# /usr/bin/env python

import vidmodex

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='onnxruntime')
warnings.filterwarnings("ignore", category=UserWarning, module='mmcv')

from tabulate import tabulate
import inspect
import yaml
import argparse
import torch
from torch import optim as thoptim
from torch.utils.data import DataLoader, Dataset
from pprint import pprint
import numpy as np
import random
import os

from vidmodex.models import Model, BlackBoxModelWrapper
from vidmodex.generator import GeneratorFactory
from vidmodex.train import train_shap_datafree
from vidmodex.tests import myprint, batch_test as test, shap_prob_test as shap_test
from vidmodex.loss import ShapLoss
from vidmodex.utils.dataloader import DatasetFactory
from vidmodex.utils.load_weight import use_pretrained, fetch_subpart
from vidmodex.utils import TensorTransformFactory
import vidmodex.config_args as config_args_main
from vidmodex.utils.arg_config import make_parser, config_update
import pytorch_lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from vidmodex.utils.callbacks import VariableAdjustmentCallback
# from pl_bolts.callbacks import BatchGradientVerificationCallback
# from vidmodex.utils.callbacks import CheckBatchGradient
from dotmap import DotMap

class ClassWiseTrainIterDataset(Dataset):

    def __init__(self, size, num_classes=1, interleave=True):
        series = torch.arange(start=0, end=num_classes).reshape(-1, 1)
        if interleave:
            self.data = series.repeat_interleave(size[0],dim=0).reshape(size[0]*num_classes, size[1])
        else:
            self.data = series.repeat(size[0],1).reshape(size[0]*num_classes, size[1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index,...], self.data[index,...]
        

class ClassWiseBlackBoxDataModule(LightningDataModule):
    def __init__(self, data_config, config_args):
        super().__init__()
        self.data_config = data_config
        self.inter = {}
        for k in dir(config_args):
            if k.startswith("_"):continue
            v = getattr(config_args, k)
            if isinstance(v, (float, int, str, list, dict, tuple)):
                self.inter[k] = v

        self.config_args = DotMap(self.inter)

    def setup(self, stage=None):
        self.train_data = ClassWiseTrainIterDataset((self.config_args.epoch_itrs, 1), self.data_config["target_dataset"]["num_classes"], interleave=True)
        val_clone_transform = TensorTransformFactory.get(f'img2{self.data_config["model"]["clone"]["name"]}')
        self.val_data = DatasetFactory.get(self.data_config["val_dataset"]["dataset_type"])(**self.data_config["val_dataset"]["init_kwargs"], transform=val_clone_transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config_args.epoch_itrs//self.config_args.dist_n, drop_last=True)

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data,batch_size=self.config_args.batch_size//self.config_args.dist_n, shuffle= True, drop_last=True, num_workers=8)
        return val_loader 

class LitModelGroup(LightningModule):
    def __init__(self,Victim, Student, Generator, Discriminator, data_config, config_args):
        super().__init__()
        # self.save_hyperparameters()
        # self.example_input_array = torch.ones(config_args.batch_size, 1)
        self.data_config = data_config
        self.inter = {}
        for k in dir(config_args):
            if k.startswith("_"):continue
            v = getattr(config_args, k)
            if isinstance(v, (float, int, str, list, dict, tuple)):
                self.inter[k] = v

        self.config_args = DotMap(self.inter)
        self.victim_max_evals = self.data_config["shap"]["max_evals"]
        self.best_acc = 0
        self.teacher = Victim(**self.data_config["model"]["victim"]["model_kwargs"])
        if self.data_config["model"]["victim"]["weight_uri"]:
            self.teacher.load_state_dict(
                fetch_subpart(
                    torch.hub.load_state_dict_from_url(
                        self.data_config["model"]["victim"]["weight"],
                    ), self.data_config["model"]["victim"].get("subpart", "full"),
                    self.data_config["model"]["victim"].get("remap_keys", None)
                ))
            
        elif self.data_config["model"]["victim"]["weight"] is not None:
            self.teacher.load_state_dict(fetch_subpart(
                torch.load(
                    self.data_config["model"]["victim"]["weight"],
                ), self.data_config["model"]["victim"].get("subpart", "full"),
                self.data_config["model"]["victim"].get("remap_keys", None)
            ))
        
        self.teacher.eval()
        self.teacher = BlackBoxModelWrapper(self.teacher, logits=self.data_config["shap"]["logits"], topk=self.data_config["model"]["victim"]["topk"])
        self.Ttransform_victim = TensorTransformFactory.get(f'gan2{self.data_config["model"]["victim"]["name"]}')
        self.Ttransform_clone = TensorTransformFactory.get(f'gan2{self.data_config["model"]["clone"]["name"]}')
        self.student = Student(**self.data_config["model"]["clone"]["model_kwargs"])

        dummy_input = torch.ones(*self.data_config["shap"]["input_shape"])
        self.shap_loss = ShapLoss(model_call=self.teacher.forward, input_sample=dummy_input, model_type=self.data_config["shap"]["model_type"], explainer_batch_size=self.data_config["shap"]["batch_size"], max_evals=self.victim_max_evals)

        self.generator = Generator(**self.data_config["model"]["generator"]["model_kwargs"])
        self.discriminator = Discriminator(**self.data_config["model"]["discriminator"]["model_kwargs"])
        
        self.val_prob_flag = self.data_config["shap"]["prob_validate"]
        self.automatic_optimization = False
        
    def training_step(self, batch, batch_idx):
        cls_idx, _ = batch
        optimizer_S, optimizer_G, optimizer_D = self.optimizers()
        
        loss_S, loss_G, loss_D, prob_loss = train_shap_datafree(self, self.config_args, teacher=self.teacher, teacher_transform=self.Ttransform_victim,
                               student=self.student, student_transform=self.Ttransform_clone, generator=self.generator,
                               discriminator=self.discriminator, shap_loss=self.shap_loss, cls_id=cls_idx, victim_max_evals=self.victim_max_evals,
                               device=self.device, optimizer=[optimizer_S, optimizer_G, optimizer_D], epoch=self.current_epoch, epoch_iters=len(cls_idx))
        
        if self.config_args.scheduler != "none":
            self.scheduler_S.step()
            self.scheduler_G.step()
        
        self.logger.experiment.add_scalar("Loss Student", loss_S, global_step=self.global_step)
        self.logger.experiment.add_scalar("Loss Generator", loss_G, global_step=self.global_step)
        self.logger.experiment.add_scalar("Loss Discriminator", loss_D, global_step=self.global_step)
        self.logger.experiment.add_scalar("Loss Prob", prob_loss, global_step=self.global_step)
        # return loss_S, loss_G
        

    def validation_step(self, batch, batch_idx):
        acc, test_loss = test(self.config_args, batch, student=self.student, generator=self.generator,
                device=self.device)
        
        if self.val_prob_flag:
            shap_prob_loss = shap_test(self.config_args, batch, self.discriminator, self.shap_loss, self.device)
            acc.update({"shap_prob_loss": shap_prob_loss})
        # torch.cuda.empty_cache()
        return {"loss": test_loss, **acc}

    def validation_epoch_end(self, validation_step_outputs):
        outputs = [[] for _ in validation_step_outputs[0].keys()]

        for out in validation_step_outputs:
            for i, v in enumerate(out.keys()):
                outputs[i].append(out[v])

        for i, v in enumerate(validation_step_outputs[0].keys()):
            self.logger.experiment.add_scalar(f"Testing {v}", np.array(outputs[i]).mean(), global_step=self.global_step)
        
        acc = 0.0
        test_loss = 0.0
        acc_flag = False
        loss_flag = False
        for i,v in enumerate(validation_step_outputs[0].keys()):
            if "acc" in v and not acc_flag:
                acc = np.array(outputs[i]).mean()
                acc_flag = True
            if "loss" in v and not loss_flag:
                test_loss = np.array(outputs[i]).mean()
                loss_flag = True
                
        file = open(self.config_args.log_file, "w")
        file.seek(0,2)
        myprint('\nTest set: Average loss: {:.4f}, Accuracy: ({:.4f}%)\n'.format(
            test_loss, 
            100*acc) , file)
        file.close()
        with open(self.config_args.log_dir + "/accuracy.csv", "a") as f:
            f.write("%d,%f\n"%(self.current_epoch, acc))
        data_set = self.data_config["target_dataset"]["name"]
        clone_name = self.data_config["model"]["clone"]["name"]
        gen_name = self.data_config["model"]["generator"]["name"]
        dis_name = self.data_config["model"]["discriminator"]["name"]
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.student.state_dict(
            ), f"{self.config_args.log_dir}/checkpoints/best_{data_set}-{clone_name}.pth")
            torch.save(self.generator.state_dict(
            ), f"{self.config_args.log_dir}/checkpoints/best_{data_set}-{gen_name}.pth")
            torch.save(self.discriminator.state_dict(
            ), f"{self.config_args.log_dir}/checkpoints/best_{data_set}-{dis_name}.pth")
        if self.config_args.store_checkpoints:
            torch.save(self.student.state_dict(), self.config_args.log_dir +
                        f"/checkpoints/clone-{clone_name}.pth")
            torch.save(self.generator.state_dict(), self.config_args.log_dir +
                        f"/checkpoints/generator-{gen_name}.pth")
            torch.save(self.discriminator.state_dict(), self.config_args.log_dir +
                        f"/checkpoints/discriminator-{dis_name}.pth")
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        
        if self.config_args.clone_optimizer == "sdg":
            optimizer_S = thoptim.SGD(self.student.parameters(),
                            lr=self.config_args.lr_S, weight_decay=self.config_args.weight_decay, momentum=0.9)
        else:
            optimizer_S = thoptim.Adam(self.student.parameters(), lr=self.config_args.lr_S)
            
        if self.config_args.gen_optimizer == "sgd":
            optimizer_G = thoptim.SGD(self.generator.parameters(
            ), lr=self.config_args.lr_G, weight_decay=self.config_args.weight_decay, momentum=0.9)
        else:
            optimizer_G = thoptim.Adam(self.generator.parameters(), lr=self.config_args.lr_G)

        if self.config_args.dis_optimizer == "sgd":
            optimizer_D = thoptim.SGD(self.discriminator.parameters(
            ), lr=self.config_args.lr_D, weight_decay=self.config_args.weight_decay, momentum=0.9)
        else:
            optimizer_D = thoptim.Adam(self.discriminator.parameters(), lr=self.config_args.lr_D)


        if self.config_args.scheduler == "multistep":
            self.scheduler_S = thoptim.lr_scheduler.MultiStepLR(
                optimizer_S, self.config_args.steps, self.config_args.scale)
            self.scheduler_G = thoptim.lr_scheduler.MultiStepLR(
                optimizer_G, self.config_args.steps, self.config_args.scale)
        elif self.config_args.scheduler == "cosine":
            self.scheduler_S = thoptim.lr_scheduler.CosineAnnealingLR(
                optimizer_S, self.config_args.number_epochs)
            self.scheduler_G = thoptim.lr_scheduler.CosineAnnealingLR(
                optimizer_G, self.config_args.number_epochs)
        ### How to fix multiple optimizers
        return optimizer_S, optimizer_G, optimizer_D


def lit_shap_blackbox_main(Victim, Student, Generator, Discriminator, data_config):
    experiment_name = data_config["experiment"]
    logger = TensorBoardLogger("runs", name=experiment_name)
    
    config_args_main.query_budget *= 10 ** 6
    config_args_main.query_budget = int(config_args_main.query_budget)
    
    pprint( { k: getattr(config_args_main,k) for k in dir(config_args_main) }, width=80)
    if logger.save_dir is not None:
        config_args_main.log_dir = logger.log_dir
        

    os.makedirs(config_args_main.log_dir, exist_ok=True)
    os.makedirs(config_args_main.log_dir + "/weights", exist_ok=True)
    # os.makedirs(config_args.log_dir + "/weights/black", exist_ok=True)
    
    if config_args_main.store_checkpoints:
        os.makedirs(config_args_main.log_dir + "/checkpoints", exist_ok=True)
        

    pytorch_lightning.utilities.seed.seed_everything(config_args_main.seed)
    np.random.seed(config_args_main.seed)
    random.seed(config_args_main.seed)
    
    kwconfig_args = {}

    global file
    student_name = data_config["model"]["clone"]["name"]
    model_dir = config_args_main.log_dir + "/checkpoints/black_box/student_{}".format(student_name)
    config_args_main.model_dir = model_dir
    if (not os.path.exists(model_dir)):
        os.makedirs(model_dir)

    config_args_main.log_file = "{}/logs.txt".format(config_args_main.model_dir)
    
    config_args_main.num_classes = data_config["target_dataset"]["num_classes"]

    # Cost calculation needs a fix
    config_args_main.cost_per_iteration = config_args_main.batch_size_z * \
                              (config_args_main.g_iter * (config_args_main.grad_m + 1) + config_args_main.d_iter)
    cost_per_epoch_shap = lambda max_evals: max_evals * config_args_main.batch_size_z * config_args_main.shap_iter 
    
    shap_cost = 0
    prev_step = 0
    init_max_evals = data_config["shap"]["max_evals"]
    for step in config_args_main.max_eval_steps:
        shap_cost += (step - prev_step) * cost_per_epoch_shap(init_max_evals)
        prev_step = step
        init_max_evals = int(config_args_main.max_eval_gamma * init_max_evals)
        if init_max_evals < config_args_main.max_eval_thresh:
            init_max_evals = 0
            break
        
    config_args_main.shap_cost_per_iteration = int(shap_cost)
    config_args_main.shap_cost_per_epoch_per_eval = config_args_main.batch_size_z * config_args_main.shap_iter 
    
    number_epochs = config_args_main.query_budget // (
            (config_args_main.cost_per_iteration + config_args_main.shap_cost_per_iteration) * config_args_main.epoch_itrs) + 1
    config_args_main.number_epochs = number_epochs
    print("\nTotal budget:", {config_args_main.query_budget // 10**6}, "M")
    print("Average Shap Cost Per iteration:", config_args_main.shap_cost_per_iteration)
    print("Shap Cost Per eval:", config_args_main.shap_cost_per_epoch_per_eval)
    print("Cost per iterations: ", config_args_main.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    steps = sorted([int(step * number_epochs) for step in config_args_main.steps])
    config_args_main.steps = steps
    print("Learning rate scheduling at steps: ", steps)


    model = LitModelGroup(Victim, Student, Generator, Discriminator, data_config=data_config, config_args=config_args_main)
    victim = data_config["model"]["victim"]["name"]
    clone = data_config["model"]["clone"]["name"]
    dataset = data_config["target_dataset"]["name"]
    print(tabulate([[f"\n\n\t\tShap based Training with {victim} as the Victim, {clone} as the Clone, {dataset} as the Target Dataset\t\t\n\n"]], tablefmt="fancy_grid"))
    
   
    
    hyperparam = {}
    for k in dir(config_args_main):
        if k.startswith("_"):continue
        v = getattr(config_args_main, k)
        if isinstance(v, (float, int, str, list, dict, tuple)):
            hyperparam[k] = v
            
    def recursive_log(config_dict, prefix=""):
        hyperdict = {}
        for k in config_dict.keys():
            if isinstance(config_dict[k], dict):
                hyperdict.update(recursive_log(config_dict[k], prefix=prefix+k+"_"))
            elif isinstance(config_dict[k], (float, int, str, list, tuple)):
                hyperdict[prefix+k] = config_dict[k]
        return hyperdict
    
    hyperparam.update(recursive_log(data_config))
    
    logger.log_hyperparams(hyperparam)    
    
    config_args_main.model = model

    data = ClassWiseBlackBoxDataModule(data_config=data_config, config_args=config_args_main)
    
    ## grad check
    # verification = BatchGradientVerificationCallback()
    # verification = CheckBatchGradient()
    variable_max_evals = VariableAdjustmentCallback("victim_max_evals", config_args_main.number_epochs,
                            config_args_main.max_eval_steps, gamma=config_args_main.max_eval_gamma, 
                            threshold=config_args_main.max_eval_thresh)
    
    trainer = Trainer(
        max_epochs=config_args_main.number_epochs,
        check_val_every_n_epoch = config_args_main.val_every_epoch,
        log_every_n_steps=config_args_main.log_every_epoch,
        callbacks = [TQDMProgressBar(refresh_rate=1), variable_max_evals], #, verification],
        accelerator=config_args_main.accelerator,
        devices=config_args_main.devices,
        num_nodes=config_args_main.num_nodes,
        resume_from_checkpoint=config_args_main.resume_ckpt,
        logger=logger
    )
    #trainer.validate(model, data)
    trainer.fit(model, data)

    print("Best Acc=%.6f" % model.best_acc)
    
    trainer.validate(model, data)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=f"{os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))}/vidmodex/configs/shap_kin400_tgan_victim_swint_clone_vivit.yaml")
    
    parser = make_parser(parser, config_args_main)
    
    parsed_args = parser.parse_args()
    
    custom_config = config_update(parsed_args, config_args_main, experiment_name="shap_blackbox_extraction")
            
    Victim          = Model.get(custom_config["model"]["victim"]["name"])
    Student         = Model.get(custom_config["model"]["clone"]["name"])
    Generator       = GeneratorFactory.get(custom_config["model"]["generator"]["name"])
    Discriminator   = GeneratorFactory.get(custom_config["model"]["discriminator"]["name"])
    lit_shap_blackbox_main(Victim, Student, Generator, Discriminator, custom_config)
