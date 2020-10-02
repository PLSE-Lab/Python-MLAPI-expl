#!/usr/bin/env python
# coding: utf-8

# # What's parquet file
# 
# I have not use with `parquet` file and do not know how to use it.
# 
# For now, I want to retrieve image information.

# # Data info
# 
# https://www.kaggle.com/c/bengaliai-cv19/data
# 
# (train/test).parquet
# Each parquet file contains tens of thousands of 137x236 grayscale images. The images have been provided in the parquet format for I/O and space efficiency. Each row in the parquet files contains an image_id column, and the flattened image.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        fullpath = os.path.join(dirname, filename)
        print('{}:{} MB'.format(fullpath, round(os.path.getsize(fullpath) / (1024.0 ** 2), 1)))

# Any results you write to the current directory are saved as output.


# # Load data
# 
# It takes a long time to read because the data size is large.

# In[ ]:


train_image_0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')


# # Data size

# In[ ]:


train_image_0.shape


# # Row info
# 
# `137 * 236 = 32332`
# 
# 0 ~ 255 grayscale data
# 
# with `image_id`

# In[ ]:


train_image_0.head()


# # Image num
# 50210

# In[ ]:


train_image_0.index


# # Remove image_id.

# In[ ]:


train_image_data = train_image_0.drop('image_id', axis=1)


# In[ ]:


train_image_data.head()


# # Create image data

# In[ ]:


IMAGE_ROW = 137 
IMAGE_COLUMN = 236

img0 = train_image_data[0:1].values[0].reshape([IMAGE_ROW, IMAGE_COLUMN])
img0


# # Show image

# In[ ]:


from matplotlib import pylab as plt

plt.imshow(img0)

