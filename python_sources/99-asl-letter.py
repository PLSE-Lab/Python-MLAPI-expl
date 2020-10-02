#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from cv2 import *
from fastai import *
from fastai.vision import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[14]:


path = Path("../input")
training_path = path/"asl_alphabet_train/asl_alphabet_train"
testing_path = path/"asl_alphabet_test/asl_alphabet_test"
import string
classes = list(string.ascii_uppercase)
classes.append(['del','space','nothing'])


# In[12]:


data = ImageDataBunch.from_folder(path, 
                                  train=training_path,
                                  valid_pct=0.2,
                                  test = testing_path,
                                  ds_tfms=get_transforms(), size=224, num_workers=6).normalize(imagenet_stats)


# In[13]:


data.show_batch(rows=3, figsize=(7,8))


# In[18]:


learn = cnn_learner(data, models.resnet50, metrics= [accuracy, error_rate], model_dir="/tmp/model/")


# In[19]:


learn.fit_one_cycle(1)


# In[20]:


learn.save('stage-1')


# In[22]:


learn.path = Path("/kaggle/working")
learn.export()


# In[23]:


learn.unfreeze()


# In[24]:


learn.lr_find()
learn.recorder.plot()


# In[25]:


learn.fit_one_cycle(2, max_lr=slice(6e-6,8e-6))


# In[26]:


learn.save('stage-2')
learn.export("stage-2-model")


# In[35]:


data = ImageDataBunch.from_folder(path, 
                                  train=training_path,
                                  valid_pct=0.2,
                                  test = testing_path,
                                  ds_tfms=get_transforms(), size=224*2, num_workers=3).normalize(imagenet_stats)


# In[36]:


learn = cnn_learner(data, models.resnet50, metrics= [accuracy, error_rate], model_dir="/tmp/model/").load('stage-2')

