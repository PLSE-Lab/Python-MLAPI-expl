#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

import os
print(os.listdir("../input/cell_images/cell_images/"))


# In[3]:


path=Path("../input/cell_images/cell_images/")
path


# In[4]:


help(get_transforms)


# In[5]:


tfms = get_transforms(
    flip_vert=True, 
    max_lighting=0.1, 
    max_zoom=1.1, 
    max_warp=0.1,
    p_affine=0.75,
    p_lighting=0.75
)


# In[6]:


src = ImageList.from_folder(path).split_by_rand_pct().label_from_folder()
src


# In[7]:


def get_data(src, size, bs, tfms):
    return (src.transform(tfms, size=size)
        .databunch(bs=bs)
        .normalize())


# In[8]:


data = get_data(src, 64, 128, tfms)


# In[9]:


data.show_batch(rows=3, figsize=(7,6))


# In[10]:


learn = cnn_learner(data, models.resnet18, metrics=[accuracy], callback_fns=[ShowGraph], model_dir="/tmp/model/")


# In[11]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[12]:


learn.fit_one_cycle(5,1e-3)


# In[13]:


learn.save('stage-1')

learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[14]:


learn.unfreeze()
learn.fit_one_cycle(5,1e-5)


# In[18]:


learn.data = get_data(src, 256, 128, tfms)

learn.freeze()
learn.fit_one_cycle(5,1e-3)


# In[19]:


learn.unfreeze()
learn.fit_one_cycle(5,1e-5)


# In[15]:


interp = ClassificationInterpretation.from_learner(learn)


# In[16]:


interp.plot_top_losses(9, figsize=(15,11))


# In[17]:


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)

