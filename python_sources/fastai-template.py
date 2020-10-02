#!/usr/bin/env python
# coding: utf-8

# Remember to turn on GPU and Internet in the settings tabs.

# In[1]:


# settings
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load libraries
from fastai import *
from fastai.vision import *

import pandas as pd


# ### Load data
# If you download data from internet : Remember to turn on the Internet settings

# In[3]:


size = 16 # ssize of input images
bs = 64 # batch size
tfms = get_transforms()


# How to load data the right way : [Link](https://docs.fast.ai/data_block.html)

# In[4]:


# Note : Try download data from internet


# In[ ]:


# Download data
path = untar_data(URLs.CIFAR); path.ls()


# In[ ]:


# Load data to DataBunch
data = ImageDataBunch.from_folder(path,train='train',test='test',valid_pct=.2,
                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)
data


# In[ ]:


data.show_batch(rows=3)


# ### Create your learner

# In[ ]:


model = models.densenet121


# In[ ]:


learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])


# In[ ]:


learn.summary()


# ### Training begin

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-2


# In[ ]:


learn.fit_one_cycle(9,slice(lr))


# In[ ]:


learn.save("stage-1")


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-4


# In[ ]:


learn.fit_one_cycle(9,slice(lr/1e2,lr))


# In[ ]:


learn.fit(10)


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


learn.save('stage-2')


# ### Training stage 2

# In[ ]:


learn.load('stage-2')


# In[ ]:


size = 24


# In[ ]:


# train with the change in images size
data = ImageDataBunch.from_folder(path,train='train',test='test',valid_pct=.2,
                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)
data


# In[ ]:


learn.data = data


# In[ ]:


learn.freeze()


# In[ ]:


lr = 1e-3


# In[ ]:


learn.fit_one_cycle(12,slice(lr))


# In[ ]:


learn.fit(5)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit(9)


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


learn.save('stage-3')


# ### Training stage 3

# In[ ]:


learn.load('stage-3')


# In[ ]:


size = 32


# In[ ]:


# train with the change in images size
data = ImageDataBunch.from_folder(path,train='train',test='test',valid_pct=.2,
                                 ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)
data


# In[ ]:


learn.data = data


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-3


# In[ ]:


learn.fit_one_cycle(12,slice(1))


# In[ ]:


learn.fit(5)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit(9)


# In[ ]:


accuracy(*learn.TTA())


# In[ ]:


# Interpretation


# In[ ]:





# In[ ]:




