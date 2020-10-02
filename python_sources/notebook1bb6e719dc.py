#!/usr/bin/env python
# coding: utf-8

# ORB Features.
# 
# Naive implementation:
# - Just get all ORB features and classify based on folder...
# - Do same on whole image from whole training set.
# 
# Simple... And a start..
# 
# 
# Any alternatives to SIFT?
# 
# Pipeline needs to be:
# - Detect fish and extract from training, label them
# - Build and SVM
# 
# Test:
# - Detect fish
# - Extract it and pass into SVM for classification

# In[15]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import ndimage
import cv2

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


Try to ID one of the fish examples


# In[21]:


#http://docs.opencv.org/trunk/d1/d89/tutorial_py_orb.html
#img_rows, img_cols= 350, 425
im_array = cv2.imread('../input/train/LAG/img_00091.jpg',0)
plt.imshow(im_array, cmap='gray')

img = im_array#cv2.imread('simple.jpg',0)
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2)

