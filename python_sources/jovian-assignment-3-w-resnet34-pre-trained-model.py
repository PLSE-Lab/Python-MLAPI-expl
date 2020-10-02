#!/usr/bin/env python
# coding: utf-8

# # Lesson 1 - What's your pet

# Welcome to lesson 1! For those of you who are using a Jupyter Notebook for the first time, you can learn about this useful tool in a tutorial we prepared specially for you; click `File`->`Open` now and click `00_notebook_tutorial.ipynb`. 
# 
# In this lesson we will build our first image classifier from scratch, and see if we can achieve world-class results. Let's dive in!
# 
# Every notebook starts with the following three lines; they ensure that any edits to libraries you make are reloaded here automatically, and also that any charts or images displayed are shown in this notebook.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import collections
get_ipython().run_line_magic('matplotlib', 'inline')


# We import all the necessary packages. We are going to work with the [fastai V1 library](http://www.fast.ai/2018/10/02/fastai-ai/) which sits on top of [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163). The fastai library provides many useful functions that enable us to quickly and easily build neural networks and train our models.

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# If you're using a computer with an unusually small GPU, you may get an out of memory error when running this notebook. If this happens, click Kernel->Restart, uncomment the 2nd line below to use a smaller *batch size* (you'll learn all about what this means during the course), and try again.

# In[ ]:


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# ## Looking at the data

# We are going to use the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) by [O. M. Parkhi et al., 2012](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf) which features 12 cat breeds and 25 dogs breeds. Our model will need to learn to differentiate between these 37 distinct categories. According to their paper, the best accuracy they could get in 2012 was 59.21%, using a complex model that was specific to pet detection, with separate "Image", "Head", and "Body" models for the pet photos. Let's see how accurate we can be using deep learning!
# 
# We are going to use the `untar_data` function to which we must pass a URL as an argument and which will download and extract the data.

# In[ ]:


help(untar_data)


# In[ ]:


from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


# In[ ]:


dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())


# In[ ]:


type(dataset)


# In[ ]:


torch.manual_seed(43)
val_size = 5000
train_size = len(dataset) - val_size


# In[ ]:


from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


# In[ ]:


train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)


# In[ ]:


from psutil import *
batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=cpu_count(), pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=cpu_count(), pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=cpu_count(), pin_memory=True)


# In[ ]:


type(train_loader)


# In[ ]:


data = ImageDataBunch(train_loader, val_loader, None, test_loader)


# In[ ]:


# cnn_learner() threw an error because data.c wasn't set
data.c = 10
ImageDataBunch


# ## Training: resnet34

# Now we will start training our model. We will use a [convolutional neural network](http://cs231n.github.io/convolutional-networks/) backbone and a fully connected head with a single hidden layer as a classifier. Don't know what these things mean? Not to worry, we will dive deeper in the coming lessons. For the moment you need to know that we are building a model which will take images as input and will output the predicted probability for each of the categories (in this case, it will have 37 outputs).
# 
# We will train for 4 epochs (4 cycles through all our data).

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'models')


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


interp.classes = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.most_confused(min_val=100)


# In[ ]:


learn.unfreeze # at this point I unlocked the whole model.
learn.fit_one_cycle(10)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=100)


# ## Results

# Hard problems stay hard. The model gains were in getting better at solving the easier classifications.
