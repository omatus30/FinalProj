import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
from PIL import Image

import torchvision.datasets as dset
import torchvision.transforms as T
import chest_xray_code.data.xrays as preprocess_dataset
import chest_xray_code.data.raw_reports as utils
import os
import torch.nn.functional as F


import numpy as np





class XrayLoader(Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 preload=False):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        filenames = glob.glob(osp.join(root, '*.png'))
        i = 0
        for fn in filenames:
            self.filenames.append(fn) # (filename, label) pair
            i +=1
            if i == 500:
                break
        if preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
        for image_fn in self.filenames:            
            image = Image.open(image_fn)
            self.images.append(image.copy())
            image.close()

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            image = self.images[index]
        else:
            image_fn = self.filenames[index]
            image = Image.open(image_fn)
        if self.transform is not None:
            image = self.transform(image)
        return image[:,100:300,100:300] 

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    

