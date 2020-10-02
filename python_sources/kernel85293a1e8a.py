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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from fastai import *
from fastai.vision import *


# In[4]:


bs = 64


# In[5]:


print(torch.cuda.is_available(), torch.backends.cudnn.enabled)


# In[6]:


path_model='/kaggle/working/'
path="/kaggle/input/cell_images"


# In[7]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26, num_workers=0,valid_pct=0.2)


# In[8]:


data.show_batch(rows=3, figsize=(7,6))


# In[9]:


print(data.classes)
len(data.classes),data.c


# In[10]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate,model_dir=f'{path_model}')


# In[11]:


learn.fit_one_cycle(10)


# In[12]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5)


# In[13]:


learn.lr_find()


# In[14]:


learn.recorder.plot()


# In[15]:


learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-5))


# In[16]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[17]:


interp.plot_top_losses(9, figsize=(15,11))


# In[18]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:





# In[ ]:





# In[ ]:




