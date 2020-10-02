#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[9]:


import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def display_images(X, rows=5, columns=5, cmap="gray", h=128,w=128):
    """ Utility function to display images
    """
    fig, ax = plt.subplots(rows,columns, figsize=(8,8))
    for row in range(rows):
        for column in range(columns):
            ax[row][column].imshow(X[(row*columns)+column].reshape(h,w), cmap=cmap)
            ax[row][column].set_axis_off()


# ### Background
# 
# Images in this data set are sample x-ray images from the [Opioid Detection Challenge](https://www.opioiddetectionchallenge.com/about-the-challenge/). The challenge calls for novel plans for rapid, nonintrusive detection tools that will help find illicit opioids in international mail. The sample images provided are of various dimensions. A typical convolutional neural network requires images of a particular size. Images in this dataset were resized and padded to fit 128x128 pixel grayscale images.

# ### Introduction
# Training a convolutional neural network for image classification, involves tuning model parameters such that a given input(image) maps to the correct output class (in this case positive or negative for opioids). To meet our model optimization goals we require large number of data samples. But this is usually not available.
# 
# What can we do if we do not have a large dataset to start with? 
# In this notebook we will explore data augmentation technique with Keras/Tensorflow when we have small image datasets. 

# ### Load Data
# Input data consists of 100 grayscale JPEG images 128x128pixel. Each row in CSV file has a label (0=Negative, 1=Positive) and 16384 pixel values corresponding to the input image.

# In[11]:


df = pd.read_csv("../input/sample/sample.csv")
df.head()


# ### Reconstitute input images from the dataframe
# Each image is 128x128pixel, single channel (grayscale), so we reshape the data from CSV file to reconstitute images from each row vector.

# In[12]:


X = df.iloc[:,1:].values.reshape(len(df),128,128,1)
y = df.iloc[:,0].values

X.shape


# ### Preview images
# Here are a few sample images from the dataset.

# In[13]:


display_images(X)


# ### Image Data Augmentation with Keras/Tensorflow
# We usually need large image datasets to meet our model optimization goals, but that is usually not available.  
# Small image datasets can be augmented artificially. The general idea is to randomly make small transformations (e.g.zoom,rotate,translate,flip, add gaussian noise etc) to original images to create new ones. While this technique is indispensable with small datasets, large datasets benefit from this as well. See this [notebook](https://www.kaggle.com/vinayshanbhag/keras-cnn-optimization) for details.
# 
# Keras [ImageDataGenerator](https://keras.io/preprocessing/image/#imagedatagenerator) makes it easy. Specific transformations may need to be chosen based on context. e.g. a vertical flip may not make sense when dealing with alphanumeric text, a 9 may turn into a 6. In this case, we have picked fillmode as a constant white background, as that makes sense given the images in this dataset. Other parameters may need to be carefully selected based on context.

# In[46]:


from keras.preprocessing.image import ImageDataGenerator
idg = ImageDataGenerator(
    rotation_range=30,
    zoom_range = 0.3, 
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.9,1.01],
    fill_mode='constant', cval=255
)


# Here are a few images produced by the ImageDataGenerator

# In[47]:


image_data = idg.flow(X, y,batch_size=25).next()
display_images(image_data[0])


# In[49]:


image_data[0][0].shape, image_data[1][0]


# In[ ]:




