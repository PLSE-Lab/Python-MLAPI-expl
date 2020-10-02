#!/usr/bin/env python
# coding: utf-8

# Bee pollen is a ball or pellet of field-gathered flower pollen packed by worker honeybees, and used as the primary food source for the hive.

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

import os
print(os.listdir("../input/pollendataset/PollenDataset/"))
pd.read_csv("../input/pollendataset/PollenDataset/pollen_data.csv").head()


# In[ ]:


path=Path("../input/pollendataset/PollenDataset/")
data = ImageDataBunch.from_csv(path, folder='images/',csv_labels='pollen_data.csv',fn_col=1, label_col=2,
                                  ds_tfms=get_transforms(flip_vert=True,do_flip=True, max_warp=0,  max_rotate=0, p_affine=0.5, max_zoom=1.1),
                                  bs=64, 
                                  num_workers=0).normalize(imagenet_stats)


# In[ ]:


print(f'Classes: \n {data.classes}')
data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir="/tmp/model/")


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10,1e-2)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.load('stage-1')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(5e-5,5e-3 ))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)


# In[ ]:




