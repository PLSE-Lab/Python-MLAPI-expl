#!/usr/bin/env python
# coding: utf-8

# # Import libraries
# 
# Typical machine learning pipeline
# 
# Data Processing -> Model -> Training -> Validation -> Prediction
# 
# We first have to view and understand the data, extract and transform it into a form that can be used for machine learning
# 
# We will use 
# - opencv to load the images as an array
# - use numpy to manipulate into a large tensor that we can train and validate
# 
# The output will be 2 large matrix X and Y

# In[ ]:


# The original code comes from https://github.com/ardamavi/Sign-Language-Digits-Dataset

import os
from os import listdir
import glob
import numpy as np
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt


# # Dataset
# 
# First open the folder sign_lang_dataset and explore the input images. It gives you a sense of what the images look like and how it is organized
# 
# The sign language images are for the digits 0 to 9 and the images are organized in directory. The directories are labeled with the digit of the images.
# 
# Let's first explore the dataset located into sign_lang_dataset/Dataset

# In[ ]:


dataset_path = "../input/sign_lang_dataset/Dataset"

sub_dir = glob.glob(os.path.join(dataset_path, '*'))
sub_dir


# ## View sample images
# 
# OpenCV by default loads images using BGR causing a strange tinge when displayed in notebook.
# 
# You need to adjust the colour mapping.
# 
# Look at cv2.cvtColor. We want to convert from BGR to RGB
# 
# [OpenCV colorspaces](https://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html)
# 

# In[ ]:



def display_img(img_path):
    img = cv2.imread(img_path)
    #TODO: colour correct the image
    color_corrected = None

    plt.imshow(color_corrected)
    plt.title(img_path)
    plt.show()


# Look at python list operations 
# 
# [Python list operations](https://www.tutorialspoint.com/python/python_lists_data_structure.htm)

# In[ ]:


img_dir = sub_dir[0]

img_files = listdir(img_dir)
img_files[:3]


# In[ ]:


display_img(img_path=os.path.join(img_dir, img_files[0]))


# In[ ]:


img_dir = sub_dir[4]

img_files = listdir(img_dir)
img_files[:3]

#TODO: Can you display the names of the 5th to 10th image files?


# In[ ]:


display_img(img_path=os.path.join(img_dir, img_files[0]))


# ## Extract images as array
# 
# In order to train the model with the images, we need to extract the pixel values as an array
# To make things simpler and manageable, we will resize the image to 64 x 64 and only use grayscale values 
# so that we only have 1 channel to deal with
# 
# Use the same function for converting colour spaces as we did earlier for converting from BGR to RBG. This time we want to resize and then convert from BGR to GRAY.
# 
# For resize please look at 
# [cv2 resize](https://docs.opencv.org/3.4.3/da/d6e/tutorial_py_geometric_transformations.html)
# Use the function that takes the image and a tuple (W, H)

# In[ ]:



def get_gsimg(image_path):
    img = cv2.imread(image_path)
    #TODO: resize the image to 64 x 64 and extract the greyscale values of the given image
    resize_img = None
    gs_img = None
    return gs_img


# In[ ]:


gs_img = get_gsimg(os.path.join(img_dir, img_files[0]))
# the shape of the image array should be (64, 64)
gs_img.shape


# In[ ]:


plt.imshow(gs_img, cmap='gray')
plt.show()


# ## Extract the images array and also constract the label array
# 
# The images is to be extracted as multi-dimensional array
# 
# For example if there are 2 images, we should end up having an 3 dimensional array that looks like this
# 
# \[
# \[\[64\], \[64\]\],
# \[\[64\], \[64\]\]
# \]
# 
# To normalize X, remember GRAYSCALE images contain pixels and each pixel value is between 0 and 255. How would you normalize it.
# 
# To convert to one-hot vector, there's a convenient function already imported to do that. Look in the imports.
# 
# 

# In[ ]:


def extract_array(dataset_path):
    label_dirs = glob.glob(os.path.join(dataset_path, '*'))
    num_classes = len(label_dirs)
    X = []
    Y = []
    for label_path in label_dirs:
        label = int(str.split(label_path, '/')[-1])
        imgs = glob.glob(os.path.join(label_path, '*.JPG'))
        for img in imgs:
            gs_img = get_gsimg(img)
            X.append(gs_img)
            Y.append(label)
    #TODO: normalize the values in X 
    X = np.array(X).astype('float32')
    #TODO: make the label a one-hot vector for example if the value is 3, then [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    Y = None
    return X, Y


# In[ ]:


X, Y = extract_array(dataset_path)
print(X.shape)
# X shape should be (2062, 64, 64)
print(Y.shape)
# Y shape should be (2062, 10)


# ## Save the arrays to files X.npy and Y.npy
# 
# The following section will not run because we cannot write to the file system.
# If you want to run this, you will have to download this and run it as a notebook in your laptop

# array_path = os.path.join(dataset_path, '..', 'Arrays')
# try:
#     os.mkdir(array_path)
# except FileExistsError:
#     print(array_path + ' already exist.')
#     
# np.save(os.path.join(array_path, 'X.npy'), X)
# np.save(os.path.join(array_path, 'Y.npy'), Y)
