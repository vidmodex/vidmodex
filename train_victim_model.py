import os
import numpy as np
import random
from tabulate import tabulate
import argparse
import inspect

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from vidmodex.models import Model, BlackBoxModelWrapper
from vidmodex.utils import TensorTransformFactory
from vidmodex.utils.dataloader import DatasetFactory
import vidmodex.config_args as config_args
from vidmodex.utils.arg_config import make_parser, config_update

import pytorch_lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from dotmap import DotMap


class TargetDataloader(LightningDataModule):
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
        transform = TensorTransformFactory.get(f'img2{self.data_config["model"]["victim"]["name"]}')

        self.train_data = DatasetFactory.get(self.data_config["train_dataset"]["dataset_type"])(**self.data_config["train_dataset"]["init_kwargs"], transform=transform)
        self.val_data = DatasetFactory.get(self.data_config["val_dataset"]["dataset_type"])(**self.data_config["val_dataset"]["init_kwargs"], transform=transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config_args.batch_size//self.config_args.dist_n, shuffle= True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data,batch_size=self.config_args.batch_size_val//self.config_args.dist_n, shuffle= True, drop_last=True, num_workers=8) 

class VictimModelGroup(LightningModule):
    def __init__(self, Victim, data_config, config_args):
        
        super().__init__()
        self.data_config = data_config
        self.inter = {}
        for k in dir(config_args):
            if k.startswith("_"):continue
            v = getattr(config_args, k)
            if isinstance(v, (float, int, str, list, dict, tuple)):
                self.inter[k] = v

        self.config_args = DotMap(self.inter)
        
        self.victim = Victim(**self.data_config["model"]["victim"]["model_kwargs"])
        

    def forward(self, x):
        return self.victim(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.victim(x)
        y_hat = F.softmax(y_logits, dim=1)
        if self.config_args.victim_train_loss == "cross_entropy":
            loss = F.cross_entropy(y_hat, y)
        else:
            raise NotImplementedError("Only cross_entropy loss is supported")
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.logger.experiment.add_scalar("train_acc", acc, self.global_step)
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.victim(x)
        y_hat = F.softmax(y_logits, dim=1)
        if self.config_args.victim_train_loss == "cross_entropy":
            loss = F.cross_entropy(y_hat, y)
        else:
            raise NotImplementedError("Only cross_entropy loss is supported")
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.logger.experiment.add_scalar("val_loss", loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc, self.global_step)
        return {"acc": acc, "loss": loss}

    def configure_optimizers(self):
        if self.config_args.victim_train_optimizer == "adam":
            optimizer = torch.optim.Adam(self.victim.parameters(), lr=self.config_args.lr_V)
        else:
            optimizer = torch.optim.SGD(self.victim.parameters(), lr=self.config_args.lr_V, momentum=self.config_args.momentum, weight_decay=self.config_args.weight_decay)

        if self.config_args.victim_train_scheduler == "multistep":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.config_args.victim_train_steps, self.config_args.victim_train_scale)
        elif self.config_args.victim_train_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config_args.number_epochs)
        
        return optimizer
    def train_dataloader(self):
        return self.dataloader
    
def victim_target_train(Victim, data_config):
    experiment_name = f'{data_config["model"]["victim"]["name"]}_target_{data_config["train_dataset"]["name"]}'
    logger = TensorBoardLogger("runs", name=experiment_name)
    
    config_args.log_dir = logger.log_dir
    # if config_args.store_checkpoints:
    #     os.makedirs(config_args.log_dir + "/checkpoints", exist_ok=True)
        
    pytorch_lightning.utilities.seed.seed_everything(config_args.seed)
    np.random.seed(config_args.seed)
    random.seed(config_args.seed)

    model = VictimModelGroup(Victim, data_config=data_config, config_args=config_args)
    victim = data_config["model"]["victim"]["name"]
    dataset = data_config["train_dataset"]["name"]
    print(tabulate([[f"\n\n\t\tTraining Victim model: {victim} for {dataset} as the Training Dataset\t\t\n\n"]], tablefmt="fancy_grid"))
    
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
    
    data = TargetDataloader(data_config=data_config, config_args=config_args)
    
    trainer = Trainer(
        strategy = DDPStrategy(find_unused_parameters=False),
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
    trainer.fit(model, data)
    
    trainer.validate(model, data)
    
    trainer.save_checkpoint(config_args.log_dir + f"/{victim}_{dataset}_{str(config_args.number_epochs/1000)}k.ckpt")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=f"{os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))}/vidmodex/configs/dfme_kin400_tgan_victim_swint_clone_vivit.yaml")
    parser.add_argument("--number-epochs", type=int, default=1000, help="Number of epochs to train the model")

    parser = make_parser(parser, config_args)
    
    parsed_args = parser.parse_args()
    
    custom_config = config_update(parsed_args, config_args)

    Victim          = Model.get(custom_config["model"]["victim"]["name"])
    
    victim_target_train(Victim, custom_config)    