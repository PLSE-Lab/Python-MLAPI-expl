#!/usr/bin/env python
# coding: utf-8

# ## Hi, everyone! 
# # Just see the images and let it sink in...
# 
# #### Credits for the public starter to @inversion and one more guy who did PCA

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
from pathlib import Path
import multiprocessing as mp

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.data import imread
from sklearn.ensemble import RandomForestClassifier
import time
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


input_path = Path('../input')
train_path = input_path / 'train'
test_path = input_path / 'test'


# In[ ]:


cameras = os.listdir(train_path)

train_images = []
for camera in cameras:
    for fname in sorted(os.listdir(train_path / camera)):
        train_images.append((camera, fname))

train = pd.DataFrame(train_images, columns=['camera', 'fname'])
print(train.shape)


# In[ ]:


test_images = []
for fname in sorted(os.listdir(test_path)):
    test_images.append(fname)

test = pd.DataFrame(test_images, columns=['fname'])
print(test.shape)


# # Lets see training images

# In[ ]:


random_images = train.sample(100)
import matplotlib.pyplot as plt

for i, r in random_images[:6].iterrows():
    x = cv2.imread("../input/train/" + r['camera'] + '/' + r['fname'])
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16,16))
    plt.imshow(x)
    plt.title(r['fname'])
    plt.show()


# # Lets see test images 

# In[ ]:



random_test_images = test.sample(100)
for i, r in random_test_images[:20].iterrows():
    x = cv2.imread("../input/test/" + r['fname'])
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(x)
    plt.title(r['fname'])
    plt.show()


# ### That's it! Thank you for reaching to the end and welcome to share your thoughts.
