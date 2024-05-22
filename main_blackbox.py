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
from vidmodex.train import train_datafree
from vidmodex.tests import myprint, batch_test as test
from vidmodex.utils.dataloader import DatasetFactory
from vidmodex.utils.load_weight import use_pretrained, fetch_subpart
from vidmodex.utils import TensorTransformFactory
import vidmodex.config_args as config_args
from vidmodex.utils.arg_config import make_parser, config_update
import pytorch_lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
# from pl_bolts.callbacks import BatchGradientVerificationCallback
# from vidmodex.utils.callbacks import CheckBatchGradient
from dotmap import DotMap


class DummyTrainIterDataset(Dataset):

    def __init__(self, size):

        self.data = torch.ones(*size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index,...], self.data[index,...]
        
class DummyBlackBoxDataModule(LightningDataModule):
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
        self.train_data = DummyTrainIterDataset((self.config_args.epoch_itrs, 1))
        val_clone_transform = TensorTransformFactory.get(f'img2{self.data_config["model"]["clone"]["name"]}')
        self.val_data = DatasetFactory.get(self.data_config["val_dataset"]["dataset_type"])(**self.data_config["val_dataset"]["init_kwargs"], transform=val_clone_transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config_args.epoch_itrs//self.config_args.dist_n, drop_last=True)

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data,batch_size=self.config_args.batch_size//self.config_args.dist_n, shuffle= True, drop_last=True, num_workers=8)
        return val_loader 

class LitModelGroup(LightningModule):
    def __init__(self,Victim, Student, Generator, data_config, config_args):
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
        self.teacher = BlackBoxModelWrapper(self.teacher, topk=self.data_config["model"]["victim"]["topk"])
        self.Ttransform_victim = TensorTransformFactory.get(f'gan2{self.data_config["model"]["victim"]["name"]}')
        self.Ttransform_clone = TensorTransformFactory.get(f'gan2{self.data_config["model"]["clone"]["name"]}')
        self.student = Student(**self.data_config["model"]["clone"]["model_kwargs"])


        self.generator = Generator(**self.data_config["model"]["generator"]["model_kwargs"])
        self.automatic_optimization = False
        
    def training_step(self, batch, batch_idx):
        e_iter, _ = batch
        optimizer_S, optimizer_G = self.optimizers()
        
        loss_S, loss_G = train_datafree(self, self.config_args, teacher=self.teacher, teacher_transform=self.Ttransform_victim,
                               student=self.student, student_transform=self.Ttransform_clone, generator=self.generator,
                               device=self.device, optimizer=[optimizer_S, optimizer_G], epoch=self.current_epoch, epoch_iters=len(e_iter))
        
        if self.config_args.scheduler != "none":
            self.scheduler_S.step()
            self.scheduler_G.step()
        
        self.logger.experiment.add_scalar("Loss Student", loss_S, global_step=self.global_step)
        self.logger.experiment.add_scalar("Loss Generator", loss_G, global_step=self.global_step)

        # return loss_S, loss_G
        

    def validation_step(self, batch, batch_idx):
        acc, test_loss = test(self.config_args, batch, student=self.student, generator=self.generator,
                device=self.device)
        
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
        for i,v in enumerate(validation_step_outputs[0].keys()):
            if "acc" in v and not acc_flag:
                acc = np.array(outputs[i]).mean()
                acc_flag = True
            if "loss" in v:
                test_loss = np.array(outputs[i]).mean()
        
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
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.student.state_dict(
            ), f"{self.config_args.log_dir}/checkpoints/best_{data_set}-{clone_name}.pth")
            torch.save(self.generator.state_dict(
            ), f"{self.config_args.log_dir}/checkpoints/best_{data_set}-{gen_name}.pth")
        if self.config_args.store_checkpoints:
            torch.save(self.student.state_dict(), self.config_args.log_dir +
                        f"/checkpoints/clone-{clone_name}.pth")
            torch.save(self.generator.state_dict(), self.config_args.log_dir +
                        f"/checkpoints/generator-{gen_name}.pth")
        self.logger.experiment.add_scalar("Testing Acc", acc, global_step=self.global_step)
        self.logger.experiment.add_scalar("Testing Loss", test_loss, global_step=self.global_step)
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
        return optimizer_S, optimizer_G


def lit_blackbox_main(Victim, Student, Generator, data_config):
    experiment_name = data_config["experiment"]
    logger = TensorBoardLogger("runs", name=experiment_name)
    
    config_args.query_budget *= 10 ** 6
    config_args.query_budget = int(config_args.query_budget)
    
    pprint( { k: getattr(config_args,k) for k in dir(config_args) }, width=80)
    if logger.save_dir is not None:
        config_args.log_dir = logger.log_dir
        

    os.makedirs(config_args.log_dir, exist_ok=True)
    os.makedirs(config_args.log_dir + "/weights", exist_ok=True)
    # os.makedirs(config_args.log_dir + "/weights/black", exist_ok=True)
    
    if config_args.store_checkpoints:
        os.makedirs(config_args.log_dir + "/checkpoints", exist_ok=True)
        

    pytorch_lightning.utilities.seed.seed_everything(config_args.seed)
    np.random.seed(config_args.seed)
    random.seed(config_args.seed)
    
    kwconfig_args = {}

    global file
    student_name = data_config["model"]["clone"]["name"]
    model_dir = config_args.log_dir + "/checkpoints/black_box/student_{}".format(student_name)
    config_args.model_dir = model_dir
    if (not os.path.exists(model_dir)):
        os.makedirs(model_dir)

    config_args.log_file = "{}/logs.txt".format(config_args.model_dir)
    
    config_args.num_classes = data_config["target_dataset"]["num_classes"]

    # Cost calculation needs a fix
    config_args.cost_per_iteration = config_args.batch_size_z * \
                              (config_args.g_iter * (config_args.grad_m + 1) + config_args.d_iter)
    number_epochs = config_args.query_budget // (
            config_args.cost_per_iteration * config_args.epoch_itrs) + 1
    config_args.number_epochs = number_epochs
    print("\nTotal budget:", {config_args.query_budget // 1000}, "k")
    print("Cost per iterations: ", config_args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    steps = sorted([int(step * number_epochs) for step in config_args.steps])
    config_args.steps = steps
    print("Learning rate scheduling at steps: ", steps)


    model = LitModelGroup(Victim, Student, Generator, data_config=data_config, config_args=config_args)
    victim = data_config["model"]["victim"]["name"]
    clone = data_config["model"]["clone"]["name"]
    dataset = data_config["target_dataset"]["name"]
    print(tabulate([[f"\n\n\t\tTraining with {victim} as the Victim, {clone} as the Clone, {dataset} as the Target Dataset\t\t\n\n"]], tablefmt="fancy_grid"))
    
   
    
    hyperparam = {}
    for k in dir(config_args):
        if k.startswith("_"):continue
        v = getattr(config_args, k)
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
    
    config_args.model = model

    data = DummyBlackBoxDataModule(data_config=data_config, config_args=config_args)
    
    ## grad check
    # verification = BatchGradientVerificationCallback()
    # verification = CheckBatchGradient()
    
    
    trainer = Trainer(
        max_epochs=config_args.number_epochs,
        check_val_every_n_epoch = config_args.val_every_epoch,
        log_every_n_steps=config_args.log_every_epoch,
        callbacks = [TQDMProgressBar(refresh_rate=1)], #, verification],
        accelerator=config_args.accelerator,
        devices=config_args.devices,
        num_nodes=config_args.num_nodes,
        resume_from_checkpoint=config_args.resume_ckpt,
        logger=logger
    )
    #trainer.validate(model, data)
    trainer.fit(model, data)

    print("Best Acc=%.6f" % model.best_acc)
    
    trainer.validate(model, data)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=f"{os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))}/vidmodex/configs/dfme_kin400_tgan_victim_swint_clone_vivit.yaml")
    
    parser = make_parser(parser, config_args)
    
    parsed_args = parser.parse_args()
    
    custom_config = config_update(parsed_args, config_args, experiment_name="blackbox_extraction")
            
    Victim          = Model.get(custom_config["model"]["victim"]["name"])
    Student         = Model.get(custom_config["model"]["clone"]["name"])
    Generator       = GeneratorFactory.get(custom_config["model"]["generator"]["name"])
    
    lit_blackbox_main(Victim, Student, Generator, custom_config)
