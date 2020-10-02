#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Creating the folder to load the pre-trained weights
get_ipython().system('mkdir -p /root/.cache/torch/checkpoints')
get_ipython().system('cp ../input/resnet50/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth')


# In[ ]:


# Importing the required dependencies

from fastai.vision import *
from fastai.metrics import error_rate
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = '/kaggle/input/painting-vs-photograph-classification-dataset'

#ImageDataBunch is a method to make a pytorch dataloader from folder
data = ImageDataBunch.from_folder(path, size=224, bs=16, 
                                  ds_tfms=get_transforms()
                                 ).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,7))


# In[ ]:


print(data.classes) # Printing the distinct classes
len(data.classes), data.c 


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate) 
# You can try with different pretrained models

learn.model


# For more information about the method is given in the following [link](http://)

# In[ ]:


#Training the model using the fit one cycle approach
learn.fit_one_cycle(4)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(5e-6 ,2e-5))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:




