#!/usr/bin/env python
# coding: utf-8

# ## Introduction

# Following code reads in all images from the train_jpg and test_jpg folder. <br>
# All the images provided are not of the same dimension. Thus, the code decides the maximum dimension and uniformly pads each image with zeros. These images are further concatenated into a single array of dimension (n_samples, x_max , y_max , 3) <br>
# n_samples  : # of images in the dataset <br>
# x_max         : Height of each padded image <br>
# y_max         :  Width of each padded image <br>
# ###### *Currently the code is concatenating only 10 images*

# ## Loading Libraries

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import glob
import zipfile

# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))


# ## Padding images and concatenate them to numpy array

# ##### Function to get the maximum dimensions of the input data location

# In[7]:


def get_maximum_dimensions(archive):
    # Get list of images from the directory
    image_paths = archive.namelist()[0:10]
    image_paths = filter(lambda x: '.jpg' in x, image_paths)
    # Set the maximum dimension to which each image needs to be padded
    data = []
    [data.append(np.array(Image.open(archive.extract(i)).convert('RGB'))) for i in image_paths]
    data = np.array(data)
    dimensions = []
    [dimensions.append(i.shape) for i in data]
    dimensions = pd.DataFrame(dimensions)
    x_max = max(dimensions.iloc[:,0])
    y_max = max(dimensions.iloc[:,1])
    return(x_max,y_max,data, image_paths)


# ##### Function to get the padded images and a concatenated array for the data

# In[10]:


def create_image_array(x_max,y_max,data):
    # Zero pad images and obtain an array to work upon
    data_final = []
    for i in data:
        left_pad = int((x_max - i.shape[0])/2)
        right_pad = x_max - i.shape[0] - int((x_max - i.shape[0])/2)
        top_pad = int((y_max - i.shape[1])/2)
        bottom_pad = y_max - i.shape[1] - int((y_max - i.shape[1])/2)
        data_final.append(np.pad(i , pad_width = ((left_pad,right_pad),(top_pad,bottom_pad),(0,0)),mode = 'constant',constant_values = 0))
    data_final = np.array(data_final)
    return(data_final)


# In[4]:


train_archive = zipfile.ZipFile('../input/train_jpg.zip', 'r')
test_archive = zipfile.ZipFile('../input/test_jpg.zip', 'r')


# In[11]:


## Get the dimensions of the images
train_x_max, train_y_max, train_data, image_paths_train = get_maximum_dimensions(train_archive)
test_x_max, test_y_max, test_data, image_paths_train = get_maximum_dimensions(test_archive)
x_max = max(train_x_max,test_x_max)
y_max = max(train_y_max,test_y_max)


# In[12]:


## Pad the images and get create the numpy array
train_images = create_image_array(x_max,y_max,train_data)
test_images = create_image_array(x_max,y_max,test_data)


# In[13]:


## Dimensions of the images
print(train_images.shape)
print(test_images.shape)


# In[14]:


## Plotting the train images to view the padded images
plt.imshow(train_images[0,:,:,:])


# In[ ]:


## Plotting the test images to view the padded images
plt.imshow(test_images[0,:,:,:])

