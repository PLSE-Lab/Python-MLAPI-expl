#!/usr/bin/env python
# coding: utf-8

# The Malaria Cell Image Dataset contains cell images categorized as Parasitezed - Uninfected.
# 
# I have added a list of other kernels which made I referred later and made improvements to this kernel.
# 
# 
# **Table of Contents**:
# 
# - Imports
# - Preparing Data
# - Explore Data
# - Model
#     - ResNet34 (Validation Accuracy of 96%)
# - Evaluating Results
# - Commit History
# - Awesome Kernals to Learn From

# # Imports

# In[1]:


# Fast AI
from fastai import *
from fastai.vision import *

# Basic Numeric Computation
import numpy as np
import pandas as pd

# Looking at directory
import os
from pathlib import Path


# # Properties

# In[2]:


commit_no = 5 # Helpful for naming output file
epochs = 4 # Increase when commiting

# Properties
base_dir = Path("../input")
model_dir = '/tmp/models'
working_dir = Path('/kaggle/working')
data = slice(10)

print(os.listdir(base_dir))

# Training Folders
malaria = base_dir/os.listdir(base_dir)[0]
print(malaria)


# # Preparing Data

# The reason for using Datablock API (https://docs.fast.ai/data_block.html) is because there was no distinct train folder so I was unable to use to_folder factory method. But I later found that I could have set train = '.' which would have worked well.
# 
# I am using seperate cells for each step so that method prompts (intellisense: shift+tab) can pop up when coding and make things easier.

# In[3]:


il = ImageList.from_folder(malaria) # There is a image list. It's in a folder


# In[4]:


il[0]


# In[6]:


il_split = il.split_by_rand_pct() # randomly split it by some percentage (default 0.2)


# In[7]:


il_labeled = il_split.label_from_folder() # label according to folder name (Parasitized; Uninfected)


# In[32]:


tfms = get_transforms(
    do_flip=True, 
    flip_vert=True, 
    max_rotate=90.0, 
    max_zoom=0, 
    max_lighting=0, 
    max_warp=0, 
    p_affine=0.75, 
    p_lighting=0.75,
)

il_transformed = il_labeled.transform(tfms,size=100, padding_mode = 'zeros') # transform them and make them the same size


# In[33]:


data = il_transformed.databunch(no_check=True, num_workers=8) # make a databunch
data = data.normalize(imagenet_stats) # normalize it with the data it was trained on so that model will converge faster


# In[34]:


def _plot(i,j,ax):
    x,y = data.train_ds[3]
    x.show(ax, y=y)

plot_multi(_plot, 3, 3, figsize=(8,8))


# ## Explore the data

# In[ ]:


data.c


# In[ ]:


data.classes


# In[ ]:


data.show_batch(3, figsize=(7,6))


# Observation: Data it seems parasitized cells have purple dots in them.

# In[ ]:


data


# In[ ]:


data.classes


# # Model

# In[ ]:


learner = cnn_learner(data, models.resnet34, metrics=[accuracy], model_dir=model_dir)


# In[ ]:


learner.fit_one_cycle(epochs)


# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(epochs)


# In[ ]:


learner.unfreeze()


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(epochs, 0.0001)


# In[ ]:


learner.freeze()


# In[ ]:


learner.fit_one_cycle(epochs, 0.0001)


# In[ ]:


learner.save(working_dir/f'resnet34_{commit_no}')


# # Evaluate Results

# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)
losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(9, figsize=(15,14), heatmap = False)


# ## Commit Messages
# 
# **v1**: Malaria Dataset ImageList.from_folder
# 
# **v4**: Made Kernel Public
# 
# **v5**: Making things presentable
# 
# **v6**: Adding more data augmentation. 
#     - Rotating would not change class
#     - Flipping won't change class

# ## Awesome Kernels to Learn From 
# 
# 
# https://www.kaggle.com/ingbiodanielh/malaria-detection-with-fastai-v1
# 
# https://www.kaggle.com/walacedatasci/malaria-detection-with-fastai-v1
