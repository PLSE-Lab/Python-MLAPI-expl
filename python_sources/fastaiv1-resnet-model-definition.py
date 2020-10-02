#!/usr/bin/env python
# coding: utf-8

# Hi,
# this is my workbook on [Lecture 7 ipynb](https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson7-resnet-mnist.ipynb) from FastAI course-v3.
# I simply edited it in order to better understand how it works, so credit goes to FastAI.  
# Mainly this notebook is about how to implement a custom model with FastAI, specifically a ResNet style model will be implemented.
# 
# If you are interested in other edited FastAI ipynb, you can find another one here: [fastaiv1 Image Classifier](https://www.kaggle.com/gianfa/fastaiv1-image-classifier)

# In[33]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *


# In[35]:


get_ipython().system('ls ../input')


# ### Data Loading

# In[36]:


path = untar_data(URLs.MNIST)
path.ls()


# In[37]:


il = ImageList.from_folder(path, convert_mode='L')
il


# In[38]:


print(il.items[0])
defaults.cmap='binary' # <-- try to comment if you want to see a different colormap
il[0].show() # <- you can access the item by index


# In[39]:


sd = il.split_by_folder(train='training', valid='testing')
sd


# In[40]:


(path/'training').ls()


# In[41]:


ll = sd.label_from_folder()
ll


# In[42]:


x,y = ll.train[0] # we can access both the variables in Train set
print(y)
x


# In[43]:


tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
ll = ll.transform(tfms)
ll


# In[44]:


bs = 128
data = ll.databunch(bs=bs).normalize()


# In[45]:


x,y = data.train_ds[0]
print(y)
x


# In[46]:


def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8))


# In[47]:


xb, yb = data.one_batch()
print(xb.shape, yb.shape)
data.show_batch(rows=3, figsize=(5,5))


# ### Basic CNN with batchnorm

# In[48]:


def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)


# Here a new model is built, using the Sequential container of Pytorch [Sequential](https://pytorch.org/docs/stable/nn.html#sequential).

# In[53]:


nInput = 1
nOutput = 10 # <-- number of classes

model = nn.Sequential(
    conv(nInput, 8), # 14  (input, output)  <--- just defined 'conv' function
    nn.BatchNorm2d(8), # (input=prev_output)
    nn.ReLU(),

    conv(8, 16), # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),

    conv(16, 32), # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),

    conv(32, 16), # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),

    conv(16, nOutput), # 1   (input, OUTPUT)
    nn.BatchNorm2d(10), #
    Flatten()     # remove (1,1) grid
)


# In[54]:


learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
print(learn.summary())


# Now put our batch on GPU

# In[57]:


xb = xb.cuda()
xb


# Let's check that our model will give us 10 outputs from our 128 images batch

# In[60]:


model(xb).shape


# In[62]:


learn.lr_find(end_lr=100)
learn.recorder.plot(suggestion=True)


# In[63]:


learn.fit_one_cycle(3, max_lr=4.5*0.1)


# ### Refactor

# check out [conv_layer](https://docs.fast.ai/layers.html#conv_layer). It _returns a sequence of nn.Conv2D, BatchNorm and a ReLU or leaky RELU activation function_

# In[68]:


def conv2(ni, nf): return conv_layer(ni, nf, stride=2)


# In[70]:


nInput = 1
nOutput = 10 
model = nn.Sequential(
    conv2(nInput, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32), # 4
    conv2(32, 16), # 2
    conv2(16, nOutput), # 1
    Flatten()      # remove (1,1) grid
)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)


# In[71]:


learn.fit_one_cycle(10, max_lr=0.1)
learn.record.plot_lr()


# ### Resnet-ish

# Here another model is built, using the Module container of PytorchPytorch, [Module](https://pytorch.org/docs/stable/nn.html#module).

# In[72]:


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf,nf)
        self.conv2 = conv_layer(nf,nf)
        
    def forward(self, x): return x + self.conv2(self.conv1(x))


# In[73]:


help(res_block)


# During the lesson Jeremy mentions Resnet compared to DenseNet. Talking about the peculiar structure of the layers he highlights that the Resnet has a _sum_ node between output of a layer, the last of a ResBlock, and the input of that block; whilst the Densenet has a _concat_ function between the same guys.  

# In[74]:


def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))


# In[75]:


model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn.lr_find(end_lr=100)
learn.recorder.plot()


# In[76]:


learn.fit_one_cycle(12, max_lr=0.05)


# Further References:
# * [Mixed Link Networks](https://www.ijcai.org/proceedings/2018/0391.pdf)
# * http://www.telesens.co/2019/01/16/neural-network-loss-visualization/

# In[ ]:




