# /usr/bin/env python

import vidmodex
import torch
from torch import optim as thoptim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from pprint import pprint
import numpy as np
import random

import os

from vidmodex.models import ViViT
from vidmodex.models import MoViNetA5 as MoViNet
from vidmodex.models import SwinT
from vidmodex.generator import Tgan
from vidmodex.generator import RefGan
from vidmodex.train import train_datafreeGAN
from vidmodex.loss import GramLoss
from vidmodex.tests import myprint, batch_test as test
from vidmodex.utils.dataloader import FakeVideosDataset, videosDataset
from vidmodex.utils.load_weight import use_pretrained
from vidmodex.utils.videoTransforms import transform_swint as transform_victim
from vidmodex.utils.videoTransforms import transform_vivit as transform_clone
from vidmodex.utils.tensorTransforms import transform_gan2swint as Ttransform_victim
from vidmodex.utils.tensorTransforms import transform_gan2vivit as Ttransform_clone
import vidmodex.config_args as config_args

import pytorch_lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from dotmap import DotMap

class GramBlackBoxDataModule(LightningDataModule):
    def __init__(self, data_config, config_args):
        super().__init__()
        self.data_config = data_config
        #self.config_args = config_args
        self.inter = {}
        for k in dir(config_args):
            if k.startswith("_"):continue
            v = getattr(config_args, k)
            if isinstance(v, (float, int, str, list, dict, tuple)):
                self.inter[k] = v
        self.config_args = DotMap(self.inter)

    def setup(self, stage=None):
        self.kinetics400_dataset = FakeVideosDataset(
            csv_file        = self.data_config["csv_file"],
            mapper_pickle   = self.data_config["mapper_pickle"],
            dataset_pickle  = self.data_config["dataset_pickle"],
            num_classes     = self.data_config["surrogate_n_class"])
        
        self.val_data = videosDataset(
            csv_file      = self.data_config["test_csv_file"],
            root_dir      = self.data_config["test_root_dir"],
            transform     = transform_clone)
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.kinetics400_dataset, batch_size=self.config_args.batch_size_train, pin_memory=True)

        
    def val_dataloader(self):
        val_loader = DataLoader(self.val_data,batch_size=self.config_args.batch_size, shuffle= True, drop_last=True)
        return val_loader

class LitModelGroup(LightningModule):
    def __init__(self, Victim, Student, Generator, Ref_GAN, AuxLoss, data_config, config_args):
        super().__init__()
        
        self.data_config = data_config
        #self.config_args = config_args
        self.inter = {}
        for k in dir(config_args):
            if k.startswith("_"):continue
            v = getattr(config_args, k)
            if isinstance(v, (float, int, str, list, dict, tuple)):
                self.inter[k] = v

        self.config_args = DotMap(self.inter)
        self.best_acc = 0
        self.teacher = Victim()
        if self.data_config["weight_uri"]:
            self.teacher.load_state_dict(
                torch.hub.load_state_dict_from_url(
                self.data_config["teacher_weight"],
                ))
            
        else:
            if self.data_config["teacher_name"]=="swint":
                self.teacher.load_state_dict(torch.load(
                    self.data_config["teacher_weight"],
                    )["state_dict"])
            else:
                self.teacher.load_state_dict(torch.load(
                    self.data_config["teacher_weight"],
                    ))
        
        self.teacher.eval()

        self.student = Student(224, 16, self.data_config["num_classes"], 16)

        self.generator = Generator()
        if data_config["ucf_gan_weights"] is not None:
            self.generator.load_state_dict(torch.load(data_config["ucf_gan_weights"])['model_state_dict'][0])

        self.Ref_GAN = Ref_GAN

        self.AuxLoss = AuxLoss
    
        self.automatic_optimization = False
        
    def training_step(self, batch, batch_idx):
        self.ref_gen = self.Ref_GAN(self.device)
        self.aux_loss = self.AuxLoss(device=self.device)
        
        optimizer_S, optimizer_G = self.optimizers()
        if self.config_args.scheduler != "none":
            self.scheduler_S.step()
            self.scheduler_G.step()

        loss_S, loss_G = train_datafreeGAN(self.config_args, teacher=self.teacher,teacher_transform=Ttransform_victim, 
                                student=self.student, student_transform=Ttransform_clone, generator=self.generator, batch=batch, ref_generator=self.ref_gen,
                                aux_loss=self.aux_loss, device=self.device, dtype=self.dtype, optimizer=[optimizer_S, optimizer_G], epoch=self.current_epoch)   

        self.logger.experiment.add_scalar("Loss Student", loss_S, global_step=self.global_step)
        self.logger.experiment.add_scalar("Loss Generator", loss_G, global_step=self.global_step)

    def validation_step(self, batch, batch_idx):
        acc, test_loss = test(self.config_args, batch, student=self.student, generator=self.generator,
                device=self.device)

        return {"acc": acc, "loss": test_loss}

    def validation_epoch_end(self, validation_step_outputs):
        accs = []
        losses = []
        for out in validation_step_outputs:
            accs.append(out["acc"])
            losses.append(out["loss"])
        
        acc = np.array(accs).mean()
        test_loss = np.array(losses).mean()
        file = open(self.config_args.log_file, 'w')
        myprint('\nTest set: Average loss: {:.4f}, Accuracy: ({:.4f}%)\n'.format(
            test_loss, 
            100*acc) , file)
        file.close()
        with open(self.config_args.log_dir + "/accuracy.csv", "a") as f:
            f.write("%d,%f\n"%(self.current_epoch, acc))
        name = self.data_config["student_name"]
        if acc > self.best_acc:
            self.best_acc = acc
            
            torch.save(self.student.state_dict(
            ), f"{self.config_args.log_dir}/checkpoint/best_kinetics-{name}.pth")
            torch.save(self.generator.state_dict(
            ), f"{self.config_args.log_dir}/checkpoint/best_kinetics-generator.pth")
        if self.config_args.store_checkpoints:
            torch.save(self.student.state_dict(), self.config_args.log_dir +
                        f"/checkpoint/student-{name}.pth")
            torch.save(self.generator.state_dict(), self.config_args.log_dir +
                        f"/checkpoint/generator.pth")
        self.logger.experiment.add_scalar("Testing Acc", acc, global_step=self.global_step)
        self.logger.experiment.add_scalar("Testing Loss", test_loss, global_step=self.global_step)

    def configure_optimizers(self):
        
        optimizer_S = thoptim.SGD(self.student.parameters(
        ), lr=self.config_args.lr_S, weight_decay=self.config_args.weight_decay, momentum=0.9)

        if self.config_args.MAZE:
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

def lit_gram_blackbox_main(Victim, Student, Generator, Ref_GAN, AuxLoss, data_config):
    config_args.query_budget *= 10 ** 6
    config_args.query_budget = int(config_args.query_budget)

    pprint( { k: getattr(config_args,k) for k in dir(config_args) }, width=80)
    config_args.log_dir += "/black_box_gram"
    print(config_args.log_dir)
    os.makedirs(config_args.log_dir, exist_ok=True)

    if config_args.store_checkpoints:
        os.makedirs(config_args.log_dir + "/checkpoint", exist_ok=True)

    pytorch_lightning.utilities.seed.seed_everything(config_args.seed)

    torch.manual_seed(config_args.seed)
    np.random.seed(config_args.seed)
    random.seed(config_args.seed)

    kwconfig_args = {}

    global file
    model_dir = "checkpoint/black_box_gram/student_{}".format(config_args.model_id)
    config_args.model_dir = model_dir
    if (not os.path.exists(model_dir)):
        os.makedirs(model_dir)

    # file = open("{}/logs.txt".format(config_args.model_dir), "w")
    config_args.log_file = "{}/logs.txt".format(config_args.model_dir)
    # config_args.file = file
    
    config_args.normalization_coefs = None
    config_args.G_activation = "torch.tanh"

    config_args.dist_n = 8 ## 8 tpu cores increase if required

    config_args.num_classes = data_config["num_classes"]

    config_args.cost_per_iteration = config_args.batch_size * \
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


    model = LitModelGroup(Victim, Student, Generator, Ref_GAN, AuxLoss, data_config=data_config, config_args=config_args)
    print("\n\t\tTraining with {config_args.model} as a Target\n".format)

    config_args.model = model

    data = GramBlackBoxDataModule(data_config=data_config, config_args=config_args)

    logger = TensorBoardLogger("runs", name="BBG")
    trainer = Trainer(
        max_epochs=config_args.number_epochs,
        callbacks = [TQDMProgressBar(refresh_rate=1)],
        accelerator=config_args.accelerator,
        devices=config_args.devices,
        precision="16",
        logger=logger
    )
    trainer.fit(model, data)

    print("Best Acc=%.6f" % model.best_acc)


if __name__=="__main__":
    Victim    = SwinT
    Student   = ViViT
    Generator = Tgan
    Ref_GAN   = RefGan
    AuxLoss   = GramLoss 
    # ToDo: Add ABSL for collecting these args
    custom_config = {}
    custom_config["student_name"]       = "vivit"
    custom_config["teacher_name"]       = "swint"
    custom_config["num_classes"]        = 400
    custom_config["surrogate_n_class"]  = 1000
    custom_config["teacher_weight"]     = "weights/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth"
    custom_config["weight_uri"]         = False
    custom_config["csv_file"]           = "../cv_seg/ActivityNet/Crawler/Kinetics/eval_kin_400_6.csv"
    custom_config["dataset_pickle"]     = "../imagenet.pkl"
    custom_config["mapper_pickle"]      = "../class_mapper.pkl"
    custom_config["test_csv_file"]      = "../cv_seg/ActivityNet/Crawler/Kinetics/eval_kin_400_6.csv"
    custom_config["test_root_dir"]      = "../cv_seg/ActivityNet/Crawler/Kinetics/eval_kin_400_6"

    custom_config["ucf_gan_weights"]    = None
    

    lit_gram_blackbox_main(Victim, Student, Generator, Ref_GAN, AuxLoss, custom_config)
