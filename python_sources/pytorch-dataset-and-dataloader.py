#!/usr/bin/env python
# coding: utf-8

# # PyTorch Dataset and DataLoader
# 
# * **1. Introduction**
# * **2. Version Check**
# * **3. Dataset and DataLoader tutorials**
#     * 3.1 Cumtom Dataset
#     * 3.2 transform is None
#     * 3.3 take a look at the dataset
#     * 3.4 transform is ToTensor()
#     * 3.5 transform is [ToTensor(), some augmentations]
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms


# ## 2. Version check
# 
# The behaviour of ToTensor() method in torchvision was changed from 0.3.0 to 0.4.0.
# 
# In 0.4.0 version, only **torch.ByteTensor** can be divided by 255 although other tensor types are not divided automatically.
# 
# so you have to convert data type of input data to **np.uint8** of ndarray.
# 
# **BE CAREFULL!!** (because pytorch official source code don't refer to this)
# 
# official code : 
# 
# ```python
# class ToTensor(object):
#     """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
#     Converts a PIL Image or numpy.ndarray (H x W x C) in the range
#     [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
#     """
# 
#     def __call__(self, pic):
#         """
#         Args:
#             pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
#         Returns:
#             Tensor: Converted image.
#         """
#         return F.to_tensor(pic)
# 
#     def __repr__(self):
#         return self.__class__.__name__ + '()'
# ```

# In[ ]:


print(torch.__version__)


# ## 3. Dataset and DataLoader Tutorial

# ### 3.1 Custom Dataset
# 
# you have to overwrite **__len__()** and **__getitem__()** functions.
# 
# official code : 
# 
# ```python
# class Dataset(object):
#     """An abstract class representing a Dataset.
#     All other datasets should subclass it. All subclasses should override
#     ``__len__``, that provides the size of the dataset, and ``__getitem__``,
#     supporting integer indexing in range from 0 to len(self) exclusive.
#     """
# 
#     def __getitem__(self, index):
#         raise NotImplementedError
# 
#     def __len__(self):
#         raise NotImplementedError
# 
#     def __add__(self, other):
#         return ConcatDataset([self, other])
# ```
# 
# - **__init__()** : initial processes like reading a csv file, assigning transforms, ... 
# - **__len__()** : return the size of input data
# - **__getitem__()** : return data and label at orbitary index

# In[ ]:


class DatasetMNIST(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((1, 28, 28))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


# let's create dataset for loading handwritten-digits data

# ### 3.2 transform is None

# In[ ]:


train_dataset = DatasetMNIST('../input/train.csv', transform=None)


# In[ ]:


# we can access and get data with index by __getitem__(index)
img, lab = train_dataset.__getitem__(0)


# we now didn't convert numpy array.

# In[ ]:


print(img.shape)
print(type(img))


# ### 3.3 take a look at the dataset
# 
# you have to use data loader in PyTorch that will accutually read the data within batch size and put into memory.

# In[ ]:


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


# we can use dataloader as iterator by using iter() function.

# In[ ]:


train_iter = iter(train_loader)
print(type(train_iter))


# we can look at images and labels of batch size by extracting data .next() method.

# In[ ]:


images, labels = train_iter.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))


# In[ ]:


# make grid takes tensor as arg
# tensor : (batchsize, channels, height, width)
grid = torchvision.utils.make_grid(images)

plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.title(labels.numpy());


# ### 3.4 transform is ToTensor()**

# In[ ]:


class DatasetMNIST2(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


# In[ ]:


train_dataset2 = DatasetMNIST2('../input/train.csv', transform=torchvision.transforms.ToTensor())


# In[ ]:


img, lab = train_dataset2.__getitem__(0)

print('image shape at the first row : {}'.format(img.size()))


# In[ ]:





# In[ ]:


train_loader2 = DataLoader(train_dataset2, batch_size=8, shuffle=True)

train_iter2 = iter(train_loader2)
print(type(train_iter2))

images, labels = train_iter2.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))


# In[ ]:


grid = torchvision.utils.make_grid(images)

plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.title(labels.numpy());


# ### 3.5 transform is [ToTensor(), some augmentations]
# 
# transforms.* methods use some type of input data like (tensor only), (tensor or numpy), (PILimage only), so you have to consider the order of transform

# **ToTensor()**
# 
# ```python
#     """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
#     Converts a PIL Image or numpy.ndarray (H x W x C) in the range
#     [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
#     (this is only for np.uint8 type)
#     """
# ```
# 
# ToTensor() takes **PIL image** or **numpy ndarray** (both shapes are (Height, Width, Channels))
# 
# **ToPILImage**
# 
# ```python
#     """Convert a tensor or an ndarray to PIL Image.
#     Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
#     H x W x C to a PIL Image while preserving the value range.
#     Args:
#         mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
#             If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
#             1. If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
#             2. If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
#             3. If the input has 1 channel, the ``mode`` is determined by the data type (i,e,
#             ``int``, ``float``, ``short``).
#     .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
#     """
# ```
# 
# ToPILImage() takes **torch.*Tensor ( C, H, W )** or **numpy ndarray ( H, W, C )**
# 
# **RandomHorizontalFlip**
# 
# ```python
#     """Horizontally flip the given PIL Image randomly with a given probability.
#     Args:
#         p (float): probability of the image being flipped. Default value is 0.5
#     """
# ```
# 
# RandomHorizontalFlip() takes **PIL Image** only

# In[ ]:


transform = transforms.Compose([
    transforms.ToPILImage(), # because the input dtype is numpy.ndarray
    transforms.RandomHorizontalFlip(0.5), # because this method is used for PIL Image dtype
    transforms.ToTensor(), # because inpus dtype is PIL Image
])


# if you want to take data augmentation, you have to make List using **torchvision.transforms.Compose**
# 
# this function can convert some image by order within **\__call__** method.
# 
# ```python
# class Compose(object):
#     """Composes several transforms together.
#     Args:
#         transforms (list of ``Transform`` objects): list of transforms to compose.
#     Example:
#         >>> transforms.Compose([
#         >>>     transforms.CenterCrop(10),
#         >>>     transforms.ToTensor(),
#         >>> ])
#     """
# 
#     def __init__(self, transforms):
#         self.transforms = transforms
# 
#     def __call__(self, img):
#         for t in self.transforms:
#             img = t(img)
#         return img
# 
#     def __repr__(self):
#         format_string = self.__class__.__name__ + '('
#         for t in self.transforms:
#             format_string += '\n'
#             format_string += '    {0}'.format(t)
#         format_string += '\n)'
#         return format_string
#     ```

# In[ ]:


train_dataset3 = DatasetMNIST2('../input/train.csv', transform=transform)


# In[ ]:


train_loader3 = DataLoader(train_dataset3, batch_size=8, shuffle=True)

train_iter3 = iter(train_loader3)
print(type(train_iter3))

images, labels = train_iter3.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))


# In[ ]:


grid = torchvision.utils.make_grid(images)

plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.title(labels.numpy());


# you can notice that the first image is horizontally flipped.

# In[ ]:





# I haven't  understood the concepts of both dataset and DataLoader yet ...

# In[ ]:




