#!/usr/bin/env python
# coding: utf-8

# # Rapid Malaria Detection
# 
# Epidemiological surveys of malaria currently rely on microscopy, polymerase chain reaction assays (PCR) or rapid diagnostic test kits for Plasmodium infections (RDTs). This study investigated whether mid-infrared (MIR) spectroscopy coupled with deep learning could constitute an alternative method for rapid malaria screening, directly from dried human blood spots.
# 
# 
# Given health system challenges facing many low-income, malaria-endemic countries, there is particular interest in non-immunological point-of-care (POC) techniques that could be readily scaled up with minimum effort

# **What is fastai?**
# 
# It is an opensource deep learning framework built on Pytorch co-founded by Jeremy Howard and Rachel Thomas. The motto of this framework is to make deep learning accessible to professionals from different backgrounds. It contains high-level components that can easily provide state-of-the-art results in standard deep learning domains.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

#setting up our enviroment
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#importing libraries
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np


# In[ ]:


x  = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/' # reading in the data
path = Path(x)
path.ls()


# In[ ]:


np.random.seed(40)
data = ImageDataBunch.from_folder(path, train = '.', valid_pct=0.2,
                                  ds_tfms=get_transforms(), size=224,
                                  num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(3, fig_size=(7,6))


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy, callback_fns=ShowGraph)


# In[ ]:


learn.fit_one_cycle(9)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


learn.show_results()

