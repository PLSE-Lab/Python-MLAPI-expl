#!/usr/bin/env python
# coding: utf-8

# Upsample/downsample functions based on Sckit-image work weird for me. (don't know how to use mode='edge', 'constant')
# 
# Followings are numpy-based upsample/downsample functions which are compatible with those in other common kernels.
# 
# 1. Preparation
# 1. Result
#     1. Numpy.pad
#     1. Sckit-image

# # 1. Preparation

# ## Import libraries

# In[ ]:


import os
import sys
import random

import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

import cv2
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import gc
gc.collect()


# In[ ]:


# Set some parameters
im_width = 101
im_height = 101
im_chan = 1
basicpath = '../input/'
path_train = basicpath + 'train/'
path_test = basicpath + 'test/'

path_train_images = path_train + 'images/'
path_train_masks = path_train + 'masks/'
path_test_images = path_test + 'images/'


# In[ ]:


img_size_ori = 101
img_size_target = 128


# ## Load images

# In[ ]:


# Loading of training/testing ids and depths

train_df = pd.read_csv(basicpath+"train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv(basicpath+"depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)


# In[ ]:


train_df["images"] = [np.array(load_img(path_train_images+"{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]


# In[ ]:


train_df["masks"] = [np.array(load_img(path_train_masks+"{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]


# ## Define a showing image function

# In[ ]:


import cv2
from IPython.display import display, Image
def cvshow(image, format='.png', rate=255 ):
    decoded_bytes = cv2.imencode(format, image*rate)[1].tobytes()
    display(Image(data=decoded_bytes))
    return


# # 2. Results

# ## A. Numpy.pad

# ### constant

# In[ ]:


img_size_ori = 101
img_size_target = 128

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return np.pad(img, [(img_size_target-img_size_ori)//2,(img_size_target-img_size_ori)-(img_size_target-img_size_ori)//2], 'constant', constant_values=(0,0))
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return img[(img_size_target-img_size_ori)//2:img_size_ori+(img_size_target-img_size_ori)//2, (img_size_target-img_size_ori)//2:img_size_ori+(img_size_target-img_size_ori)//2]


# In[ ]:


orig_img = np.squeeze(np.array(train_df.images.tolist()).reshape(-1, 101, 101, 1)[100, :, :, :])
cvshow(orig_img)
edge_mag_img = np.squeeze(np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)[100, :, :, :])
cvshow(edge_mag_img)
cvshow(downsample(edge_mag_img))


# ### edge

# In[ ]:


img_size_ori = 101
img_size_target = 128

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return np.pad(img, [(img_size_target-img_size_ori)//2,(img_size_target-img_size_ori)-(img_size_target-img_size_ori)//2], 'edge')
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return img[(img_size_target-img_size_ori)//2:img_size_ori+(img_size_target-img_size_ori)//2, (img_size_target-img_size_ori)//2:img_size_ori+(img_size_target-img_size_ori)//2]


# In[ ]:


orig_img = np.squeeze(np.array(train_df.images.tolist()).reshape(-1, 101, 101, 1)[100, :, :, :])
cvshow(orig_img)
edge_mag_img = np.squeeze(np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)[100, :, :, :])
cvshow(edge_mag_img)
cvshow(downsample(edge_mag_img))


# ### reflect

# In[ ]:


img_size_ori = 101
img_size_target = 128

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return np.pad(img, [(img_size_target-img_size_ori)//2,(img_size_target-img_size_ori)-(img_size_target-img_size_ori)//2], 'reflect')
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return img[(img_size_target-img_size_ori)//2:img_size_ori+(img_size_target-img_size_ori)//2, (img_size_target-img_size_ori)//2:img_size_ori+(img_size_target-img_size_ori)//2]


# In[ ]:


orig_img = np.squeeze(np.array(train_df.images.tolist()).reshape(-1, 101, 101, 1)[100, :, :, :])
cvshow(orig_img)
edge_mag_img = np.squeeze(np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)[100, :, :, :])
cvshow(edge_mag_img)
cvshow(downsample(edge_mag_img))


# # B. sckit-image

# ### constant

# In[ ]:


img_size_ori = 101
img_size_target = 128

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), cval = 0, mode='constant', preserve_range=True)
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


# In[ ]:


orig_img = np.squeeze(np.array(train_df.images.tolist()).reshape(-1, 101, 101, 1)[100, :, :, :])
cvshow(orig_img)
edge_mag_img = np.squeeze(np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, 128, 1)[100, :, :, :])
cvshow(edge_mag_img)
cvshow(downsample(edge_mag_img))


# 
# ### edge

# In[ ]:


img_size_ori = 101
img_size_target = 128

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='edge', preserve_range=True)
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='edge', preserve_range=True)


# In[ ]:


orig_img = np.squeeze(np.array(train_df.images.tolist()).reshape(-1, 101, 101, 1)[100, :, :, :])
cvshow(orig_img)
edge_mag_img = np.squeeze(np.array(train_df.images.map(upsample).tolist()).reshape(-1, 128, 128, 1)[100, :, :, :])
cvshow(edge_mag_img)
cvshow(downsample(edge_mag_img))


# ### refrect

# In[ ]:


img_size_ori = 101
img_size_target = 200

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), order=0, mode='reflect', preserve_range=True)
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), order=0, mode='reflect', preserve_range=True)


# In[ ]:


orig_img = np.squeeze(np.array(train_df.images.tolist()).reshape(-1, 101, 101, 1)[100, :, :, :])
cvshow(orig_img)
edge_mag_img = np.squeeze(np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)[100, :, :, :])
cvshow(edge_mag_img)
cvshow(downsample(edge_mag_img))

