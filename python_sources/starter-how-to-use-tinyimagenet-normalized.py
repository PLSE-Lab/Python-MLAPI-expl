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


# ## Prerequisites
# 
# This dataset requires the class definitions located at https://github.com/z-a-f/zaf_funcs.git

# In[ ]:


# Get the support classes that were used to create this dataset
get_ipython().system('git clone https://github.com/z-a-f/zaf_funcs.git')


# ### Loading the set
# 
# The dataset is using the `pickle`.
# 
# **Note: The dataset is loaded into the RAM, and WILL use a lot of it to load :)**

# In[ ]:


import pickle
from zaf_funcs.transforms import *

# Load the pickles
with open('/kaggle/input/tinyimagenet-normalized/val_data.pickle', 'rb') as val_file:
    print("Loading the validation pickle...")
    val_data = pickle.load(val_file)
with open('/kaggle/input/tinyimagenet-normalized/train_data.pickle', 'rb') as train_file:
    print("Loading the training pickle...")
    train_data = pickle.load(train_file)


# ### Accessing individual samples
# 
# The dataset supports indexing, which returns the sample as a tuple `(img, lbl)`.
# However, the dataset has to be in the `.eval()` mode.
# If in training mode, you need to access batches of data (due to transformation stuff).
# The images are in the `NCHW` layout, and saved as PyTorch tensors.
# 

# In[ ]:


train_data.eval()


# In[ ]:


print(type(train_data), type(val_data))
print("Number of samples in the training set:", len(train_data))
print("Number of samples in the validation set:", len(val_data))
print("Training sample 0:", train_data[0])


# In[ ]:


import matplotlib.pyplot as plt
import torch

train_data.eval()

# Show the first sample
sample = train_data[42]
img, label = sample
img = img.permute((0, 2, 3, 1))  # NCHW -> NHWC
img = img.squeeze()

# Denormalize the image
mean = torch.tensor([0.485, 0.456, 0.406])  # From the dataset description
std = torch.tensor([0.229, 0.224, 0.225])  # From the dataset description
img = img * std
img = img + mean

print(img.min(), img.max())
plt.imshow(img)
plt.title("Label: {}".format(label.item()))


# ## Loading batches of data
# 
# You can use the `.batch_loader(batch_nums)` to load the datasets. That also allows you to use the `.train` mode on the datasets

# In[ ]:


train_data.train()

train_loader = train_data.batch_loader(16)

for imgs, lbls in train_loader:
    print("Batch img dims:", imgs.shape)
    print("Batch lbl dims:", lbls.shape)
    break


# The `.train` mode allows you to add data transformations.
# To add the transformations you need to change the `train_data.transform.ts` list.

# In[ ]:


for t in train_data.transform.ts:
    print(type(t))
for t in train_data.final_transform.ts:
    print(type(t))

