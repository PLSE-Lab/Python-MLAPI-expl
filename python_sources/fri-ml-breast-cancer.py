#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# reload modules before executing user code
get_ipython().run_line_magic('load_ext', 'autoreload')
# reload all modules every time before executing the Python code
get_ipython().run_line_magic('autoreload', '2')
# view plots in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
import shutil
import pandas as pd
from fastai.vision import *
from fastai.widgets import DatasetFormatter, ImageCleaner


# In[ ]:


path = '/kaggle/input/IDC_regular_ps50_idx5'
path


# In[ ]:


tfms = get_transforms()


# In[ ]:


data = ImageDataBunch.from_folder(path, ds_tfms=tfms, valid_pct=0.2, size=224)


# In[ ]:


data.show_batch(rows=3, figsize=(8, 8))


# In[ ]:


learner = cnn_learner(data, models.resnet34, metrics=accuracy)


# In[ ]:


learner.model_dir = '/kaggle/working/models'


# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot()


# In[ ]:


lr = 1e-03


# In[ ]:


learner.fit_one_cycle(1, lr)

