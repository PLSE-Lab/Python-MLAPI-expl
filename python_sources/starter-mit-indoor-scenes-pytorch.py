#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This starter kernal include how to read this dataset in pytorch and a simple EDA on Data. The primary dataset in the tutorial is MIT Scenes which can be downloaded from http://web.mit.edu/torralba/www/indoor.html or using the dataset attached to the kernal.
# 
# The database contains 67 Indoor categories, and a total of 15620 images. The number of images varies across categories, but there are at least 100 images per category. All images are in jpg format. The images provided here are for research purposes only.

# ## Loading common Packages
# Loading some common libraries and packages

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# Loading Pytorch Libraries

# In[ ]:


import torch
import torchvision
import torch.nn as nn


# ## Pytorch Dataset `Object`
# 
# 
# First we need to design Pytorch dataset object to hold data. This dataset object can intrect with pytorch _dataLoader_. We can see the effects in just a momemt

# In[ ]:


from PIL import Image
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

from torchvision import datasets
class Scenes(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 train=True):
        imagelist_file = 'Images.txt'
        if train:
            imagelist_file = 'Train'+imagelist_file
        else :
            imagelist_file = 'Test' + imagelist_file
        filesnames = open(os.path.join(root, imagelist_file)).read().splitlines()
        self.root = os.path.join(root, 'indoorCVPR_09/Images')
        classes, class_to_idx = find_classes(self.root)

        images = []

        for filename in list(set(filesnames)):
            target = filename.split('/')[0]
            path = os.path.join(root, 'indoorCVPR_09/Images/' + filename)
            item = (path, class_to_idx[target])
            images.append(item)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = images

        self.imgs = self.samples
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)


# There is 2 txt file in the current version of the dataset:
# * TrainImages.txt: contains the file names of each training image. Total 67x80 images
# * TestImages.txt: contains the file names of each test image. Total 67x20 images
# 

# ## Reading Data in Pytorch
# 
# 
# we are making a `dict` of datasets, test and validation set.

# In[ ]:


datasets = {x : Scenes(root='../input/', train=True if x == 'train' else False)
            for x in ['train', 'val']}
print("number of Datasets {}.".format(len(datasets)))
print('Number of Classes {}.'.format(len(datasets['train'].classes)))
for dstype in datasets:
    print('{} set contains {} images'.format(dstype, len(datasets[dstype])))


# ## Data Transforms
# 
# Data Transforms are used to preprocess the data set before using as input for ML/DL models.
# In pytorch we can define our preprocessings in `torchvision.transforms.Compose` which can take a list of transforms. Although we can define own transforms but there are some predefined and cools transforms already avaliable in the pytorch library.
# 
# *Note:* _these transforms are applied in the order of defination_

# In[ ]:


# Data transforms
from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Reloading Dataset with transforms applied on it

# In[ ]:


datasets = {x : Scenes(root='../input/', train=True if x == 'train' else False, 
                       transform=data_transforms[x])
            for x in ['train', 'val']}
class_names = datasets['train'].classes
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}


# `torch.utils.data.DataLoader` helps Datasets to itrate, also we can create batches in our dataset and define number of workers to load datasets

# In[ ]:


from torch.utils import data
dataloaders = {x: data.DataLoader(datasets[x], batch_size=4,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'val']}


# ## Visulizating Data in Batch
# 
# Science we have transformed our data to Tensors so it can't be displayed using `matplotlib` so we need to convert it back to numpy `ndarrays`

# In[ ]:


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)  # pause a bit so that plots are updated


# visulizing a batch

# In[ ]:


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
import torchvision
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# ## Conclusion
# This concludes your Reading in pytorch i hope this gives you a good idea how to load Data use data in pytorch. I'll soon be uploading a new tutorial traning a simple Nueral Network in pytorch.
# 
# Have a nice Day Happy kaggling! 
