#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's| several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system(' du -sk /kaggle/input/prostate-cancer-grade-assessment/train_images')

# about 35GB


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import openslide
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split


# In[ ]:


train_df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
test_df = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


def preprocess_image(image_path, desired_size=224):
    biopsy = openslide.OpenSlide(image_path)
    im = np.array(biopsy.get_thumbnail(size=(desired_size,desired_size)))
    im = np.resize(im,(desired_size,desired_size,3)) / 255
    
    return im


# In[ ]:


img_1 = f"../input/prostate-cancer-grade-assessment/train_images/{train_df['image_id'][25]}.tiff"
a = openslide.OpenSlide(img_1)
a.get_thumbnail(size=(512,512))


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# get the number of training images from the target\\id dataset\nN = train_df.shape[0] # run on all data(50percent of data)\n#N = 1000 # run on sample\n# create an empty matrix for storing the images\nx_train = np.empty((N, 224, 224, 3), dtype=np.float32)\n# loop through the images from the images ids from the target\\id dataset\n# then grab the cooresponding image from disk, pre-process, and store in matrix in memory\nfor i, image_id in enumerate(tqdm(train_df['image_id'])):\n    x_train[i, :, :, :] = preprocess_image(\n        f'../input/prostate-cancer-grade-assessment/train_images/{image_id}.tiff'\n    )\n    # if sampling\n    if i >= N-1:\n        break")


# In[ ]:


if os.path.exists(f'../input/prostate-cancer-grade-assessment/test_images'):
    # do the same thing as the last cell but on the test\holdout set
    N = test_df.shape[0]
    x_test = np.empty((N, 224, 224, 3), dtype=np.float32)
    for i, image_id in enumerate(tqdm(test_df['image_id'])):
        x_test[i, :, :, :] = preprocess_image(
            f'../input/prostate-cancer-grade-assessment/test_images/{image_id}.tiff'
        )


# In[ ]:


# pre-processing the target (i.e. one-hot encoding the target)
y_train = pd.get_dummies(train_df['isup_grade']).values.astype(np.int32)[0:N]

# Further target pre-processing

# Instead of predicting a single label, we will change our target to be a multilabel problem; 
# i.e., if the target is a certain class, then it encompasses all the classes before it. 
# E.g. encoding a class 4 retinopathy would usually be [0, 0, 0, 1], 
# but in our case we will predict [1, 1, 1, 1]. For more details, 
# please check out Lex's kernel.

y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[:, 5] = y_train[:, 5]

for i in range(4, -1, -1):
    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train_multi, 
    test_size=0.30, 
    random_state=2020
)


# In[ ]:


np.save('X_train.npy', x_train)
np.save('y_train.npy', y_train)


# In[ ]:


print("heelo")


# In[ ]:




