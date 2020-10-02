#!/usr/bin/env python
# coding: utf-8

# # Image Classification Models from scratch
# In this kernel, I will briefly discuss about writing Image classification models from scratch using fastaiv2 and Pytorch. Right now fastaiv2 is right now scheduled to be properly released in June 2020. If you want to check more about the new version of fast.ai [do read this paper](https://arxiv.org/abs/2002.04688)
# 
# <font size=3 color="red">Please upvote this kernel if you like it.</font>

# # Contents
# 
# * [<font size=4>Installing fastai and downloading data</font>](#1)
# * [<font size=4>Peeking at the data</font>](#2)
# * [<font size=4>Basic Model</font>](#3)
# * [<font size=4>Simplifying Things</font>](#4)
# * [<font size=4>Making a resnet kinda of model</font>](#5)
# * [<font size=4>Ending Notes</font>](#6)

# # Acknowledgment
# 
# This notebook is a fork from fastai2 megastudy group conducted by [Zach Mueller](https://www.kaggle.com/muellerzr). More details about the online study group and fantastic fastai2 resources can be found in the [Practical Deep Learning for Coders 2.0 repo](https://github.com/muellerzr/Practical-Deep-Learning-for-Coders-2.0/)

# ## Installing fastai and downloading data <a id='1'>

# In[ ]:


#Run once per session
get_ipython().system('pip install fastai2')


# In[ ]:


# grab vision related APIs
from fastai2.basics import *
from fastai2.vision.all import *
from fastai2.callback.all import *


# In[ ]:


# Download our data

path = untar_data(URLs.MNIST)


# ## Peeking at the data <a id='2'>

# In[ ]:


items= get_image_files(path)
items[0]


# In[ ]:


# create an image object with ImageBlock

img = PILImageBW.create(items[0])
img.show()


# In[ ]:


# Split our data with GrandparentSplitter, which will make use of a train and valid folder.

splits = GrandparentSplitter(train_name='training', valid_name='testing')
splits = splits(items)


# In[ ]:


# understand what it's split with?
splits[0][:5], splits[1][:5]


# In[ ]:


dsrc = Datasets(items, tfms=[[PILImageBW.create],[parent_label, Categorize]],
                   splits=splits)


# In[ ]:


show_at(dsrc.train, 3)


# In[ ]:


tfms = [ToTensor(), CropPad(size=34, pad_mode=PadMode.Zeros), RandomCrop(size=28)]
gpu_tfms = [IntToFloatTensor(), Normalize()]


# **Tranformations**
# - ToTensor: Converts to tensor
# - CropPad and RandomCrop: Resizing transforms
# - Applied on the CPU via after_item
# 
# **Enables GPU usage(gpu_tfms)**
# - IntToFloatTensor: Converts to a float
# - Normalize: Normalizes data

# In[ ]:



dls = dsrc.dataloaders(bs=128, after_item=tfms, after_batch=gpu_tfms)


# In[ ]:


dls.show_batch()


# In[ ]:


# passing as a batch
xb, yb = dls.one_batch()
xb.shape, yb.shape


# In[ ]:


# no of classes and class labels
dls.vocab


# ## Basic model <a id='3'>
# 
# Our models are made up of layers, and each layer represents a matrix multiplication to end up with our final y. For this image problem, we will use a Convolutional layer, a Batch Normalization layer, an Activation Function, and a Flattening layer

# In[ ]:


def conv(ni, nf):
    return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)


# In[ ]:


def bn(nf): return nn.BatchNorm2d(nf)


# In[ ]:


def ReLU(): return nn.ReLU(inplace=True)


# In[ ]:


# five CNN layers, 1 -> 32 -> 10

model = nn.Sequential(
            conv(1,8),
            bn(8),
            ReLU(),
            conv(8,16),
            bn(16),
            ReLU(),
            conv(16, 32),
            bn(32),
            ReLU(),
            conv(32,16),
            bn(16),
            ReLU(),
            conv(16,10),
            bn(10),
            Flatten() # and flatten it into a single dimention of predictions
)


# In[ ]:


learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.summary()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit_one_cycle(3, lr_max=1e-1)


# ## Simplifying Things <a id='4'>
# 
# ConvLayer contains a Conv2d, BatchNorm2d, and an activation function

# In[ ]:


def conv2(ni, nf):
    return ConvLayer(ni, nf, stride=2)


# In[ ]:


net = nn.Sequential(conv2(1,8),
                   conv2(8,16),
                   conv2(16,32),
                   conv2(32,16),
                   conv2(16,10),
                   Flatten())


# In[ ]:


learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=accuracy)


# In[ ]:


learn.fit_one_cycle(3, lr_max=1e-1)


# ## Making a ResNet kinda of model <a id='5'>
# 
# The ResNet architecture is built with what are known as ResBlocks. Each of these blocks consist of two ConvLayers that we made before, where the number of filters do not change. Let's generate these layers.
# 
# [A brief description of what's ResBlocks is mentioned in this kernel](https://www.kaggle.com/kurianbenoy/solving-kannada-mnist-with-fastai)

# In[ ]:


class ResBlock(Module):
    def __init__(self, nf):
        self.conv1 = ConvLayer(nf, nf)
        self.conv2 = ConvLayer(nf, nf)
  
    def forward(self, x): return x + self.conv2(self.conv1(x))


# In[ ]:


net = nn.Sequential(
    conv2(1,8),
    ResBlock(8),
    conv2(8,16),
    ResBlock(16),
    conv2(16,32),
    ResBlock(32),
    conv2(32,16),
    ResBlock(16),
    conv2(16,10),
    Flatten()
)


# In[ ]:


net


# 
# Awesome! We're building a pretty substantial model here. Let's try to make it even simpler. We know we call a convolutional layer before each ResBlock and they all have the same filters, so let's make that layer!

# In[ ]:


def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), ResBlock(nf))


# In[ ]:


net = nn.Sequential(
    conv_and_res(1,8),
    conv_and_res(8,16),
    conv_and_res(16,32),
    conv_and_res(32,16),
    conv2(16,10),
    Flatten()
)


# In[ ]:


learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(3, lr_max=1e-1)


# # Ending note <a id="6"></a>
# 
# Our final trained model with `Resnet kind of architecture` gives **99.2% accuracy**
# 
# <font size=4 color="red">This concludes my fastaiv2 kernel heavily borrowed from @muellerzr. Please upvote this kernel if you like it. It motivates me to produce more quality content :)</font>
# 
