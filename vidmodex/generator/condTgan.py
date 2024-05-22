import math
from PIL import Image
from torch.utils.data import Dataset
import cv2
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from moviepy.editor import ImageSequenceClip
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

FRAMES_PER_VIDEO = 16
NOISE_SIZE = 100
COND_VEC_SIZE = 32

class TemporalGenerator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Create a sequential model to turn one vector into 16 vectors
        self.num_classes = num_classes
        # seperate embedding layer for the content vector for temporal generation
        self.embed_layer = nn.Embedding(self.num_classes, COND_VEC_SIZE)
        
        self.model = nn.Sequential(
            nn.ConvTranspose1d(NOISE_SIZE+COND_VEC_SIZE, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, NOISE_SIZE, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # initialize weights according to paper
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose1d:
            nn.init.xavier_uniform_(m.weight, gain=2**0.5)

    def forward(self, x, cls_idx):
        # reshape x so that it can have convolutions done
        cond_vec = self.embed_layer(cls_idx)
        x = torch.cat([x, cond_vec], dim=-1)
        x = x.view(-1, NOISE_SIZE + COND_VEC_SIZE, 1)    
        # apply the model and flip the
        x = self.model(x).transpose(1, 2)
        return x

    def constrain(self):
        pass

class VideoGenerator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # instantiate the temporal generator
        self.temp = TemporalGenerator(self.num_classes)
        # seperate embedding layer for the content vector for content generation
        self.embed_layer = nn.Embedding(self.num_classes, COND_VEC_SIZE)
        # create a transformation for the temporal vectors
        self.fast = nn.Sequential(
            nn.Linear(NOISE_SIZE + COND_VEC_SIZE, 256 * 4**2, bias=False),
            nn.BatchNorm1d(256 * 4**2),
            nn.ReLU()
        )

        # create a transformation for the content vector
        self.slow = nn.Sequential(
            nn.Linear(NOISE_SIZE + COND_VEC_SIZE, 256 * 4**2, bias=False),
            nn.BatchNorm1d(256 * 4**2),
            nn.ReLU()
        )

        # define the image generator
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, 
                               stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, 
                               stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, 
                               stride=1, padding=1),
            nn.Tanh()
        )

        # initialize weights according to the paper
        self.fast.apply(self.init_weights)
        self.slow.apply(self.init_weights)
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=-0.01, b=0.01)

    def forward(self, x, cls_idx=None):
        # pass our latent vector through the temporal generator and reshape
        if cls_idx is None:
            cls_idx = torch.zeros((x.shape[0],), dtype=torch.long).to(device=x.device)
        z_fast = self.temp(x, cls_idx).contiguous()
        
        cond_vec = self.embed_layer(cls_idx)
        
        frame_len = z_fast.shape[1]
        cond_vec_fast = cond_vec.unsqueeze(1).repeat_interleave(frame_len, dim=1)
        # print(cond_vec_fast.shape,z_fast.shape)
        z_fast = z_fast.view(-1, NOISE_SIZE)
        cond_vec_fast = cond_vec_fast.view(-1, COND_VEC_SIZE)
        z_fast = torch.cat([z_fast, cond_vec_fast], dim=-1)
        # transform the content and temporal vectors
        z_fast = self.fast(z_fast).view(-1, 256, 4, 4)
        # print(x.shape, cond_vec.shape)
        x = torch.cat([x, cond_vec], dim=-1)
        z_slow = self.slow(x).view(-1, 256, 4, 4).unsqueeze(1)
        # after z_slow is transformed and expanded we can duplicate it
        z_slow = torch.cat([z_slow]*FRAMES_PER_VIDEO,
                           dim=1).view(-1, 256, 4, 4)

        # concatenate the temporal and content vectors
        z = torch.cat([z_slow, z_fast], dim=1)

        # transform into image frames
        out = self.model(z)

        return out.view(-1, FRAMES_PER_VIDEO, 3, 256, 256).transpose(1, 2) # TCHW->CTHW

    def constrain(self):
        pass

def singular_value_clip(w):
    dim = w.shape
    # reshape into matrix if not already MxN
    if len(dim) > 2:
        w = w.reshape(dim[0], -1)
    u, s, v = torch.svd(w, some=True)
    s[s > 1] = 1
    return (u @ torch.diag(s) @ v.t()).view(dim)


class VideoDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2)
        )

        self.conv2d = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)

        # initialize weights according to paper
        self.model3d.apply(self.init_weights)
        self.init_weights(self.conv2d)

    def init_weights(self, m):
        if type(m) == nn.Conv3d or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight, gain=2**0.5)

    def forward(self, x):
        h = self.model3d(x)
        # turn a tensor of R^NxTxCxHxW into R^NxCxHxW

        h = torch.reshape(h, (-1, 512, 4, 4))
        h = self.conv2d(h)
        return h

    def constrain(self):
        for module in list(self.children()):
            if type(module) == nn.Conv3d or type(module) == nn.Conv2d:
                module.weight.data = singular_value_clip(module.weight)
            elif type(module) == nn.BatchNorm3d:
                gamma = module.weight.data
                std = torch.sqrt(module.running_var)
                gamma[gamma > std] = std[gamma > std]
                gamma[gamma < 0.01 * std] = 0.01 * std[gamma < 0.01 * std]
                module.weight.data = gamma
