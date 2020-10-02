#!/usr/bin/env python
# coding: utf-8

# In[72]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[73]:


train_df = pd.read_csv('../input/boneage-training-dataset.csv')
test_df = pd.read_csv('../input/boneage-test-dataset.csv')


# In[74]:


train_df.head()


# In[75]:


import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


# In[76]:


train_img_path = Path('../input/boneage-training-dataset/boneage-training-dataset/')


# In[77]:


img = Image.open(train_img_path / (str(train_df.id.values[0]) + '.png'))
plt.imshow(img,cmap='gray')


# In[78]:


from keras.preprocessing.image import ImageDataGenerator


# In[79]:


train_img_gen = ImageDataGenerator(rescale=1./255)
train_img_set = train_img_gen.flow_from_directory(train_img_path, target_size=(100,100), class_mode=None, batch_size=10)


# In[80]:


train_df['path'] = train_df['id'].map(lambda x: '{}.png'.format(x))


# In[85]:


train_df.head()


# In[83]:


img = Image.open(train_img_path / train_df.path.values[0])
plt.imshow(img, cmap='gray')


# In[86]:


train_img_gen = ImageDataGenerator(rescale=1./255)
train_img_set = train_img_gen.flow_from_dataframe(dataframe=train_df, 
                                                  directory=train_img_path, 
                                                  x_col='path', y_col='boneage',
                                                  target_size=(256, 256), 
                                                  class_mode='other',
                                                  color_mode='grayscale',
                                                  batch_size=10)


# In[87]:


for data, label in train_img_set:
    print(data.shape)
    print(label)
    break


# In[ ]:




