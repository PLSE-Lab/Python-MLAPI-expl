#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import random
import cv2
import numpy as np # linear algebra


# In[ ]:


## constants
TRAIN_DIR = "../input/train/"
TEST_DIR = "../input/test/"
TRAIN_SIZE = 22500
TEST_SIZE = 2500
DEV_RATIO = 0.1
IMAGE_HEIGHT = IMAGE_WIDTH = 128
OUTPUT_SIZE = 2


# # Data preparation
# To start, we read provided data. 
# 
# The *../input/train/* dir contains 12500 cat images and 12500 dog images.
# Each filename contains "cat" or "dog" as label.

# In[ ]:


## tool functions
def ex_time(func):
    start_time = datetime.datetime.now()
    
    def wrapper(*args, **kwargs):
        print("start time: {}".format(start_time))
        res = func(*args, **kwargs)
        
        end_time = datetime.datetime.now()
        ex_time = end_time - start_time
        print("end time: {}".format(end_time))
        print("excute time: {} seconds".format(ex_time.seconds))

        return res
       
    return wrapper

def display(image, image_width=IMAGE_HEIGHT, image_height=IMAGE_HEIGHT, interpolation=3):
    # (784) => (28,28)
    one_image = image.reshape(image_width,image_height, interpolation)
    
    new_f = plt.figure()
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()
    plt.close()


# In[ ]:


## data utility functions
def dense_to_one_hot(labels_dense, num_classes):
    """
    # convert class labels from scalars to one-hot vectors
    # 0 => [1 0 0 0 0 0 0 0 0 0]
    # 1 => [0 1 0 0 0 0 0 0 0 0]
    # ...
    # 9 => [0 0 0 0 0 0 0 0 0 1]
    """
    num_labels = labels_dense.shape[0]
    #print("num_labels:", num_labels)
    index_offset = np.arange(num_labels) * num_classes
    #print("index_offset:", index_offset)
    labels_one_hot = np.zeros((num_labels, num_classes))
    #print("labels_one_hot:", labels_one_hot)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    #print(index_offset + labels_dense.ravel())
    #print("labels_one_hot2:", labels_one_hot)
    return labels_one_hot

def split_data(images, labels, dev_ratio=DEV_RATIO):
    dev_count = int(labels.shape[1] * DEV_RATIO)
    dev_images = images[:, :dev_count]
    train_images = images[:, dev_count:]
    dev_labels = labels[:, :dev_count]
    train_labels = labels[:, dev_count:]
    print("train images shape: {}, train labels shape:{},     dev images shape: {}, dev labels shape: {}".format(train_images.shape, train_labels.shape, dev_images.shape, dev_labels.shape))
    return train_images, train_labels, dev_images, dev_labels


# In[ ]:


#@ex_time
def pre_data(dirname=TRAIN_DIR, file_count=1000):
    all_filenames = os.listdir(dirname)
    random.shuffle(all_filenames)
    filenames = all_filenames[:file_count]
    
    ## images
    images = np.zeros((file_count, IMAGE_HEIGHT*IMAGE_WIDTH*3))
    for i in range(file_count):
        imgnd_origin = cv2.imread(dirname+filenames[i])
        imgnd_resized = cv2.resize(imgnd_origin, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
        imgnd_flatten = imgnd_resized.reshape(1,-1)
        images[i] = imgnd_flatten
    
    ## labels from filenames
    labels_list = ["dog" in filename for filename in filenames]
    labels = np.array(labels_list, dtype='int8').reshape(file_count, 1)
    
    ## shuffle
    permutation = list(np.random.permutation(labels.shape[0]))
    shuffled_labels = labels[permutation, :]
    shuffled_images = images[permutation, :]
    
    ## dense to one hot
    labels = dense_to_one_hot(shuffled_labels, OUTPUT_SIZE)
    ## normalization
    images = shuffled_images/255.0
    
    return images.T, labels.T


# In[ ]:


images, labels = pre_data(file_count=100)


# In[ ]:


train_images, train_labels, dev_images, dev_labels = split_data(images, labels)


# In[ ]:


print(train_images.shape, train_labels.shape, dev_images.shape, dev_labels.shape)


# # Analysis

# <img src="../output/mems_cpu01.jpeg" style="width: 100%">

# * too much memory
# 
# * too much cpu

# # Next

# [GraphLab Create]()
# 
#     * SFrame: It is an efficient disk-based tabular data structure which is not limited by RAM. It helps to scale analysis and data processing to handle large data set (Tera byte), even on your laptop.
#     
#     * SGraph: Graph helps us to understand networks by analyzing relationships between pair of items. Each item is represented by a vertex in the graph

# In[ ]:




