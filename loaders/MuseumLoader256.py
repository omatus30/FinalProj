import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import matplotlib.pyplot as plt
import random

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




class MuseumLoader(Dataset):
    """
    A customized data loader for MNIST.
    """
    def __init__(self,
                 root,
                 sample_size = 500,
                 transform=None,
                 preload=False):
        """ Intialize the MNIST dataset
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.sample_size = 500

        # read filenames
        filenames = glob.glob(osp.join(root, '*.jpg'))
        
        
        #randomly select filenames for more even distribution
        #import pdb; pdb.set_trace();
        file_samples =  random.sample(filenames, sample_size)
        
        for fn in file_samples:
            self.filenames.append(fn) # (filename, label) pair

 

        # if preload dataset into memory
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
            # load images
            image = Image.open(image_fn)
            # avoid too many opened files bug
            self.images.append(image.copy())
            image.close()
            #self.labels.append(label)

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
                
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
        else:
            # If on-demand data loading
            try:
                image_fn = self.filenames[index]
                image = Image.open(image_fn)
                image = image.resize((256,256),Image.ANTIALIAS)
            except IOError:
                os.remove(image_fn)
                print(image_fn)
            #image = image[:,0:200,0:200]
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
        # return image and label

                
        return image[:,:256,:256]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    

