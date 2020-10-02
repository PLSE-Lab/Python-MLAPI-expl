#!/usr/bin/env python
# coding: utf-8

# # Exploring and Preprocessing the input images
# 
# This kernel intends to explore the image dataset, and preprocess them to be 224x224 to match ImageNet, and make it compatible with most architectures in Keras or Tensorflow. The resulting data will be:
# * `X_train`: 25361x224x224x3
# * `X_test`: 7960x224x224x3
# * `y_train`: 25361x5005

# In[ ]:


import os
import cv2
import time
import math
import psutil
import multiprocessing

import numpy as np # linear algebra
from PIL import Image
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[ ]:


print(os.listdir("../input"))


# ## Exploration

# In[ ]:


label_df = pd.read_csv('../input/train.csv')
submission_df = pd.read_csv('../input/sample_submission.csv')
label_df.head()


# In[ ]:


label_df['Id'].describe()


# In[ ]:


# Display the most frequent ID (without counting new_whale)
label_df['Id'].value_counts()[1:16].plot(kind='bar')


# In[ ]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(5*columns, 3*rows))

    for i in range(columns*rows):
        image_path = df.loc[i,'Image']
        image_id = df.loc[i,'Id']
        img = cv2.imread(f'../input/train/{image_path}')
        fig.add_subplot(rows, columns, i+1)
        plt.title(image_id)
        plt.imshow(img)

display_samples(label_df)


# The width of the image seem to be bigger than the height. We will have to pad the images, then resize them to 224x224x3

# ## Preprocessing

# In[ ]:


def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
    l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
    if is_rgb:
        pad_width = ((t,b), (l,r), (0, 0))
    else:
        pad_width = ((t,b), (l,r))
    return pad_width

def pad_and_resize_cv(image_path, dataset, desired_size=224):
    img = cv2.imread(f'../input/{dataset}/{image_path}')
    
    pad_width = get_pad_width(img, max(img.shape))
    padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
    
    resized = cv2.resize(padded, (desired_size,)*2).astype('uint8')
    
    return resized

def pad_and_resize_pil(image_path, dataset, desired_size=224):
    '''Experimental'''
    im = Image.open(f'../input/{dataset}/{image_path}')
    
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    resized = im.resize(new_size)
    im_array = np.asarray(resized)
    
    pad_width = get_pad_width(im_array, desired_size)
    padded = np.pad(im_array, pad_width=pad_width, mode='constant', constant_values=0)
    
    return padded


def pad_and_resize(image_path, dataset, desired_size=224, mode='cv'):
    if mode =='pil':
        return pad_and_resize_pil(image_path, dataset, desired_size)
    else:
        return pad_and_resize_cv(image_path, dataset, desired_size)


# ### Padding process and resizing with OpenCV

# In[ ]:


img = cv2.imread(f'../input/train/{label_df.loc[0,"Image"]}')

pad_width = get_pad_width(img, max(img.shape))
padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
resized = cv2.resize(padded, (224,224))
plt.imshow(resized)


# ## Pad and resize all the images

# In[ ]:


target_dummies = pd.get_dummies(label_df['Id'])
train_label = target_dummies.columns.values
y_train = target_dummies.values
print(y_train.shape)


# In[ ]:


def process_dataset(dataset):
    resized_imgs = []
    
    if dataset == 'train':
        dataset_names = label_df["Image"]
    else:
        dataset_names = submission_df['Image']

    for image_path in dataset_names:
        resized_imgs.append(pad_and_resize(image_path, dataset))

    X = np.stack(resized_imgs)
    return X


# In[ ]:


start_time = time.time()

with multiprocessing.Pool(1) as pool: 
    X_train, X_test = pool.map(process_dataset, ["train", "test"])
    
print(f"Images loaded in {time.time() - start_time:.2f} sec")

