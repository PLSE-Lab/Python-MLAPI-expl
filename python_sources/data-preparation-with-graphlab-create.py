#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from array import array
import numpy as np
import graphlab as gl
gl.canvas.set_target('ipynb')


# In[ ]:


## constants
TRAIN_DIR = "../input/train/"
TEST_DIR = "../input/test/"
TRAIN_SIZE = 22500
TEST_SIZE = 2500
DEV_RATIO = 0.1
IMAGE_HEIGHT = IMAGE_WIDTH = 128
CHANNELS = 3
OUTPUT_SIZE = 2


# # Data preparation
# To start, we read provided data. 
# 
# The *../input/train/* dir contains 12500 cat images and 12500 dog images.
# Each filename contains "cat" or "dog" as label.

# In[ ]:


image_sframe = gl.image_analysis.load_images(TRAIN_DIR)


# In[ ]:


image_sarray = image_sframe['image']


# In[ ]:


resized_image_sarry = gl.image_analysis.resize(image_sarray, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)


# In[ ]:


img1 = resized_image_sarry[0]


# In[ ]:


img1.pixel_data


# In[ ]:


data = np.zeros((1000, IMAGE_HEIGHT*IMAGE_WIDTH*CHANNELS))
for i in range(1000):
    img = resized_image_sarry[i]
    px_data = img1.pixel_data.reshape(1, IMAGE_HEIGHT*IMAGE_WIDTH*CHANNELS)
    data[i] = px_data


# # Analysis

# <img src="../output/mems_cpu02.png" style="width: 100%">

# * less memory and cpu than cv2
# 

# # Next

# Explore TensorFlow.image
