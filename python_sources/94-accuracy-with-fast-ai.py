#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *


# In[ ]:


path = Path('../input/boats')


# In[ ]:


(path).ls()


# In[ ]:


data = ImageDataBunch.from_folder(path, train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(flip_vert=False),
                                  size=128,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')


# In[ ]:


lr = 1e-3


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-2')


# In[ ]:


data = ImageDataBunch.from_folder(path, train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(flip_vert=False),
                                  size=256,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-3))


# In[ ]:


learn.save('stage-3')


# In[ ]:




