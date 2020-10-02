#!/usr/bin/env python
# coding: utf-8

# This is the very first time I tried to build a deep learning model. if there is some error in my model or there are some areas where I can improve this model please let me know

# ## Fruit Classification Model
# This model will tell you about Fruit just by seeing there images. you can download the dataset that I used from kaggle dataset (https://www.kaggle.com/moltean/fruits). I have used a Convolutional neural network to build this model.
# 
# This project is a homework assignment for Fastai's Deep learning for coders lesson 1

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai import *
from fastai.vision import *


# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
bs = 64


# In[3]:


path = URLs.LOCAL_PATH/'../input/fruits-360_dataset/fruits-360'
tfms = get_transforms(do_flip=False)
np.random.seed(2)
data = ImageDataBunch.from_folder((path), train="Training", valid="Test", ds_tfms=tfms, size=52)
data.show_batch(rows=3, figsize=(5,5))


# ## Training: resnet50

# In[13]:


learn = create_cnn(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")


# In[14]:


learn.fit_one_cycle(4, max_lr=slice(3e-5,3e-4))


# ## Fine-tuning model

# In[15]:


learn.save("4-epoch")


# In[19]:


learn.unfreeze()


# In[20]:


learn.fit_one_cycle(6, max_lr=slice(3e-5,3e-4))


# In[21]:


learn.save("stage-1")

