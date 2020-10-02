#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# **Step-1.** Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py 

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

import torch
print('pytorch version: ',torch.__version__)
import torch.utils.data as data
import fastai
print('fastai version: ',fastai.__version__)
import torchvision.models


# **Step-2.** Setting path to data (Labels = folder_name) 

# In[ ]:


img_dir = '../input/car_data'
path = Path(img_dir)
path.ls()


# **Step-3.** Splitting data(Training set=80%, Validation set=20%), setting batch size = 64, Normalize data using imagenet_stats(transfer learning)

# In[ ]:


data = ImageDataBunch.from_folder(f'{path}',valid_pct = 0.2,size = 224,bs = 64).normalize(imagenet_stats)


# **Step-4.** Let's see how our data looks like and how many classes are there.

# In[ ]:


for classes, numbers in enumerate(data.classes[:15]):
    print(classes,':',numbers)
len(data.classes),data.c


# In[ ]:


data.show_batch(rows = 3,figsize = (15,15))


# **Step-5.** Setting ResNet-50 CNN arcitechture and training only last layers of our model****.

# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")


# In[ ]:


learn.fit_one_cycle(6)


# **Step-6.** Saving trained model

# In[ ]:


learn.save('stage-1')


# **Step-7.** Fine-Tuning. Train whole model(all layers) and save it.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(6)


# In[ ]:


learn.save('stage-2',return_path=True)


# **Step-8.** Results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(4, figsize=(14,14),heatmap=False)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


input, target = learn.get_preds()


# TOP-1 Accuracy:

# In[ ]:


print (top_k_accuracy(input=input, targs=target,k=1))


# Top-3 Accuracy:

# In[ ]:


print (top_k_accuracy(input=input, targs=target,k=3))

