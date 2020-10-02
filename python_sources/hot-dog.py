#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# #Loading of important Packages

# In[ ]:


import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook


# In[ ]:


hot_dog_image_dir = '../input/seefood/train/hot_dog'
not_hot_dog_image_dir = '../input/seefood/train/not_hot_dog'
hot_dog_test_image_dir = '../input/seefood/test/hot_dog'
not_hot_dog_test_image_dir = '../input/seefood/test/not_hot_dog'


# In[ ]:


train_dir='../input/seefood/train'
test_dir='../input/seefood/test'


# In[ ]:


train_data_hd = [os.path.join(hot_dog_image_dir, filename)
              for filename in os.listdir(hot_dog_image_dir)]
train_data_nhd = [os.path.join(not_hot_dog_image_dir, filename)
              for filename in os.listdir(not_hot_dog_image_dir)]
test_data_hd = [os.path.join(hot_dog_test_image_dir, filename)
              for filename in os.listdir(hot_dog_test_image_dir)]
test_data_nhd = [os.path.join(not_hot_dog_test_image_dir, filename)
              for filename in os.listdir(not_hot_dog_test_image_dir)]


# In[ ]:


#There total 998 images, 498 in train set and 500 in test set.
print(len(train_data_hd))
print(len(train_data_hd))
print(len(test_data_hd))
print(len(test_data_nhd))


# In[ ]:


# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# In[ ]:


# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the transforms, define the dataloaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, num_workers=4)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8, num_workers=4)


# In[ ]:


print("Length of Train Dataset: {:.1f}.. ".format(len(train_dataset)))
print("Length of Test Dataset: {:.1f}.. ".format(len(test_dataset)))


# In[ ]:


#Visualization of dataset
def imshow(imgs, title=None):
    """Imshow for Tensor."""
    imgs = imgs.numpy().transpose((1, 2, 0))
    plt.imshow(imgs)
    if title is not None:
        plt.title(title)
    


# Get a batch of training data
inputs, _ = next(iter(train_dataloader))

# Make a grid from batch
imgs = torchvision.utils.make_grid(inputs)

imshow(imgs)


# In[ ]:


train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

