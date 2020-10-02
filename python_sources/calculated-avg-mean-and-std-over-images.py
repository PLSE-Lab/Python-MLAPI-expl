#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Note: This is calculated for RGB order of channels!**

# In[ ]:


def calc_avg_mean_std(img_names, img_root):
    mean_sum = np.array([0., 0., 0.])
    std_sum = np.array([0., 0., 0.])
    n_images = len(img_names)
    for img_name in tqdm(img_names):
        img = cv2.imread(img_root + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean, std = cv2.meanStdDev(img)
        mean_sum += np.squeeze(mean)
        std_sum += np.squeeze(std)
    return (mean_sum / n_images, std_sum / n_images)


# In[ ]:


train_img_root = '../input/train_images/'
train_img_names = os.listdir(train_img_root)
train_mean, train_std = calc_avg_mean_std(train_img_names, train_img_root)
train_mean, train_std


# In[ ]:


test_img_root = '../input/test_images/'
test_img_names = os.listdir(test_img_root)
test_mean, test_std = calc_avg_mean_std(test_img_names, test_img_root)
test_mean, test_std


# **Results:**
# * For Train:
# 
# |      | Red         | Green       | Blue        |
# |------|-------------|-------------|-------------|
# | Mean | 87.68971919 | 87.68971919 | 87.68971919 |
# | Std  | 35.61160271 | 35.61160271 | 35.61160271 |
# 
# * For Test:
# 
# |      | Red         | Green       | Blue        |
# |------|-------------|-------------|-------------|
# | Mean | 66.17581321 | 66.17581321 | 66.17581321 |
# | Std  | 35.08540253 | 35.08540253 | 35.08540253 |
# 
# So, we can see that images are actually grayscale, but test images are darker than train images.

# In[ ]:


train_mean / 255., train_std / 255.


# In[ ]:


test_mean / 255., test_std / 255.


# **Values for pixels in 0..1 range:**
# * For Train:
# 
# |      | Red         | Green       | Blue        |
# |------|-------------|-------------|-------------|
# | Mean | 0.34388125 | 0.34388125 | 0.34388125 |
# | Std  | 0.13965334 | 0.13965334 | 0.13965334 |
# 
# * For Test:
# 
# |      | Red         | Green       | Blue        |
# |------|-------------|-------------|-------------|
# | Mean | 0.25951299 | 0.25951299 | 0.25951299 |
# | Std  | 0.13758981 | 0.13758981 | 0.13758981 |
