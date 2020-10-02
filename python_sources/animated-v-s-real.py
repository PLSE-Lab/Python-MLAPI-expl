#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:




from fastai import *
from fastai.vision import *


# In[6]:


bs = 10


# In[26]:


monkey_images_path = '../input/animted1/'
tfms = get_transforms()
data = ImageDataBunch.from_folder(monkey_images_path, train='training', valid='validation',ds_tfms=tfms, bs=6, size=128,classes=["an","rl"])


# In[38]:


data.valid_dl.x[4]
data.show_batch(rows=3, figsize=(15,6))


# In[28]:


path_model='/kaggle/working/'
path_input='/kaggle/input/'
learn_resnet34 = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir=f'{path_model}')


# In[29]:


learn_resnet34.fit_one_cycle(4)


# In[30]:


interp = ClassificationInterpretation.from_learner(learn_resnet34)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[35]:




interp.plot_top_losses(6, figsize=(15,11))


# In[ ]:




