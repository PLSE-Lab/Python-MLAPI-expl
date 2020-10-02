#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random

image_dir = '/kaggle/input/face-images-with-marked-landmark-points/face_images.npz'
keypoints_dir = '/kaggle/input/face-images-with-marked-landmark-points/facial_keypoints.csv'

# Load the dataset 
images = np.load(image_dir)['face_images']
facial_keypoints = pd.read_csv(keypoints_dir)

# Standardize the values of images 
images = images/255


# In[ ]:


# Get the features in the dataset 
features = facial_keypoints.columns.tolist()

# Get the index of all images with non-null values 
selection_index = np.nonzero(facial_keypoints.left_eye_center_x.notna() & facial_keypoints.right_eye_center_x.notna() & facial_keypoints.nose_tip_x.notna() & facial_keypoints.mouth_center_bottom_lip_x.notna())[0]
# Number of selections 
m = selection_index.shape[0]

image_dim = images.shape[0]

# Get the matching image samples 
image_samples = images[:,:,selection_index]
# Get the keypoints for the samples and standardize the results 
keypoints = np.zeros((m, 8))
keypoints[:,0] = facial_keypoints.left_eye_center_x[selection_index].values
keypoints[:,1] = facial_keypoints.left_eye_center_y[selection_index].values
keypoints[:,2] = facial_keypoints.right_eye_center_x[selection_index].values
keypoints[:,3] = facial_keypoints.right_eye_center_y[selection_index].values
keypoints[:,4] = facial_keypoints.nose_tip_x[selection_index].values
keypoints[:,5] = facial_keypoints.nose_tip_y[selection_index].values
keypoints[:,6] = facial_keypoints.mouth_center_bottom_lip_x[selection_index].values
keypoints[:,7] = facial_keypoints.mouth_center_bottom_lip_y[selection_index].values


# In[ ]:


# Plot a random image with facial keypoints 
index = random.randint(0, m)
plt.imshow(image_samples[:,:,index], cmap='gray')
# Add the features 
for i in range(0,8,2): 
    x_pos = keypoints[index,i]
    y_pos = keypoints[index,i+1]
    plt.plot(x_pos, y_pos, 'ro')


# In[ ]:




