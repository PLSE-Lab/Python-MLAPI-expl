#!/usr/bin/env python
# coding: utf-8

# # About
# 
# This kernel presents the [104 Flowers: Garden of Eden dataset](https://www.kaggle.com/msheriey/104-flowers-garden-of-eden) which is a JPEG conversion of the [Flower Classification with TPUs competition TFRecords dataset](https://www.kaggle.com/c/flower-classification-with-tpus). 
# 
# The aim of this kernel and dataset is to help people:
# 
# *     Practice transforming it back to TFRecords.
# *     Add it to the [Valentine's kernel](https://www.kaggle.com/mpwolke/valentine-s-day-no-tpu).
# *     View the original dataset more easily.
# *     Create EDAs.
# 
# or else.

# <font size=4 color='red'> If you find this kernel useful, please don't forget to upvote. Thank you. </font>

# # Reading Files

# In[ ]:


import os

import random

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image


# In[ ]:


DATASET_DIR = '/kaggle/input/104-flowers-garden-of-eden/jpeg-512x512'

TRAIN_DIR  = DATASET_DIR + '/train'


# In[ ]:


FLOWER_NAMES = []

FLOWER_TRAIN_FILEPATHS = {}

for root, dir_names, _ in os.walk(TRAIN_DIR):
    
    for dir_name in dir_names:
        FLOWER_NAMES.append(dir_name)
        FLOWER_TRAIN_FILEPATHS[dir_name] = []
    
        for dir_root, _, dir_filenames in os.walk(os.path.join(root, dir_name)):
            
            for filename in dir_filenames: 
                FLOWER_TRAIN_FILEPATHS[dir_name].append(os.path.join(dir_root, filename))


# # Flower Names

# In[ ]:


print('len(FLOWER_NAMES): ', len(FLOWER_NAMES))


# In[ ]:


FLOWER_NAMES


# ## Naviagation

# In[ ]:


FLOWER_TRAIN_FILEPATHS['artichoke']


# # Grow Flowers

# In[ ]:


def grow_flowers(flower_type, sample_size=5):
    flowers = [Image.open(flower).convert('RGB') for flower in random.sample(FLOWER_TRAIN_FILEPATHS[flower_type], sample_size)]
                
    n_flowers = len(flowers)
    
    figure = plt.figure()
    for i, flower in enumerate(flowers):
        figure.add_subplot(1, np.ceil(n_flowers), i + 1)
        plt.axis('off')
        plt.imshow(flower)
        
    figure.set_size_inches(np.array(figure.get_size_inches()) * n_flowers)
    
    plt.show()


# # Garden of Eden

# In[ ]:


for flower_name in FLOWER_NAMES:
    print(flower_name)
    grow_flowers(flower_name)

