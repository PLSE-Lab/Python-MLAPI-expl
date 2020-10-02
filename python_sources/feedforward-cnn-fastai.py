#!/usr/bin/env python
# coding: utf-8

# This kernel is an adaptation of https://www.kaggle.com/kenseitrg/simple-fastai-exercise

# In[1]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path
from fastai import *
from fastai.vision import *
import torch


# In[2]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


data_folder = Path("../input")
#data_folder.joinpath('train').ls()


# In[4]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")


# In[5]:


test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')
trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
train_img = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')
        .split_by_rand_pct(0.01)
        .label_from_df()
        .add_test(test_img)
        .transform(trfm, size=128)
        .databunch(path='.', bs=64, device= torch.device('cuda:0'))
        .normalize(imagenet_stats)
       )


# In[6]:


train_img.show_batch(rows=3, figsize=(7,6))


# In[7]:


learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy])


# In[8]:


#learn.lr_find()
#learn.recorder.plot()


# In[9]:


lr = 3e-02
learn.fit_one_cycle(5, slice(lr))


# In[10]:


#learn.unfreeze()
#learn.lr_find()
learn.recorder.plot()


# In[11]:


#learn.fit_one_cycle(1, slice(1e-06))


# In[12]:


#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_top_losses(9, figsize=(7,6))


# In[13]:


preds,_ = learn.get_preds(ds_type=DatasetType.Test)


# In[14]:


test_df.has_cactus = preds.numpy()[:, 0]


# In[15]:


test_df.to_csv('submission.csv', index=False)


# In[ ]:




