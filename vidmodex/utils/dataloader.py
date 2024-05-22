import os
import cv2
from PIL import Image
import numpy as np

import pandas as pd
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100, MNIST, Caltech101, Caltech256

class DatasetFactory:
    _dataset_classes = {}

    @classmethod
    def get(cls, dataset_type:str):
        try:
            return cls._dataset_classes[dataset_type]
        except KeyError:
            raise ValueError(f"unknown product type : {dataset_type}")

    @classmethod
    def register(cls, dataset_type:str):
        def inner_wrapper(wrapped_class):
            cls._dataset_classes[dataset_type] = wrapped_class
            return wrapped_class
        return inner_wrapper

class videosDataset(Dataset):

    def __init__(self, dataset_csv, dataset_root, transform=None):

        self.annotations = pd.read_csv(dataset_csv)

        self.root_dir = dataset_root

        self.transform = transform

        ll = sorted(list(self.annotations.iloc[:,1].unique()))
        self.dic = {}
        for id, i in enumerate(ll):
            self.dic[i] = id

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        vid_path = os.path.join(
            self.root_dir, (self.annotations.iloc[index, 0]))
        # print(vid_path)
        vid_label = torch.tensor(self.dic[self.annotations.iloc[index, 1]])
        # put the labels into a dictionary?

        vid, fc = self.FrameCapture(vid_path)
        # print("before",vid[:,60:-60,:,:])
        if self.transform:
            vid = self.transform(vid)
        # print("after",vid[:, :, 20:-20, :])
        # exit()
        return (vid, vid_label)

    def FrameCapture(self, path):

        # Path to video file
        vidObj = cv2.VideoCapture(path)

        # Used as counter variable
        count = 0
        success, image = vidObj.read()
        # checks whether frames were extracted
        frems = []
        
        while success:

            # vidObj object calls read
            # function extract frames
            
            frems.append(image)
            # print(image)
            success, image = vidObj.read()
            count += 1
        fc = len(frems)
        # if fc<=16:
        #     print("**dataloader.FrameCapture**",path, fc)
        
        frames = []
        if fc < 16:
            cnt = 0
            for frem in frems:
                frames.append(frem)
                cnt += 1
            while cnt<16:
                if len(frames)==0:
                    frames.append(np.zeros((640,640,3), dtype=np.uint8))
                else:
                    frames.append(np.zeros_like(frames[0]))
                cnt +=1
            frames = np.array(frames, dtype=np.uint8)
            return frames, 16
        
        for i in range(0, fc, (fc-1) // 15):

            image = frems[i]

            ma = max(image.shape[1], image.shape[0])
            h, w = image.shape[0], image.shape[1]
            image = cv2.copyMakeBorder(image, (ma - h) // 2, (ma-h)//2,
                                        (ma-w)//2, (ma-w)//2, cv2.BORDER_CONSTANT, None, value=[0, 0, 0])
            
            frames.append(image[:, :, [2, 1, 0]])
        frames = np.array(frames[:16],np.uint8)
        # from PIL import Image

        # frame = []
        # for fr in frames:
        #     frame.append(Image.fromarray(fr))
        # # save image list as a gif (ordered alphabetically by glob)
        # exit()
        # frames = frames.reshape(1, *frames.shape)
        # print("shape", frames.shape)
        return (frames, fc)

class FakeVideosDataset(Dataset):

    def __init__(self, dataset_csv, mapper_pickle=None, dataset_pickle=None, num_classes=1000):
        if dataset_pickle is not None:
            with open(dataset_pickle, 'rb') as f:
                inv_map = pickle.load(f)
            self.dataset_map = {v:k for k,v in inv_map.items()}
        else:
            self.dataset_map = None
        self.annotations = pd.read_csv(dataset_csv)
        self.sub_class = (mapper_pickle is not None)
        if self.sub_class:
            with open(mapper_pickle, 'rb') as f:
                self.mapper = pickle.load(f)
        self.sub_cls_map = {it:i  for i, it in enumerate(set([v for k,v in self.mapper.items()])) }
        ll = sorted(list(self.annotations.iloc[:,1].unique()))
        self.dic = {}
        for id, i in enumerate(ll):
            self.dic[i] = id

        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        vid_label = self.annotations.iloc[index, 1]
        vid = vid_label
        # print(vid_label)
        if self.sub_class:
            vid_label = self.mapper.get(vid_label)
            # print(vid_label)
        if self.dataset_map is None:
            return vid_label, self.sub_cls_map[vid_label]
        else:
            return vid_label, self.dataset_map[vid_label], self.sub_cls_map[vid_label]

DatasetFactory.register('videoDataset')(videosDataset)
DatasetFactory.register('imagenet')(ImageNet)
DatasetFactory.register('cifar10')(CIFAR10)
DatasetFactory.register('cifar100')(CIFAR100)
DatasetFactory.register('mnist')(MNIST)
DatasetFactory.register('caltech101')(Caltech101)
DatasetFactory.register('caltech256')(Caltech256)
DatasetFactory.register('fakeVideoDataset')(FakeVideosDataset)