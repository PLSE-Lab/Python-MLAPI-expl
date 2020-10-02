#!/usr/bin/env python
# coding: utf-8

# # Library

# In[ ]:


import numpy as np
import pandas as pd
import cv2
from keras.preprocessing import image
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[ ]:


#settings
pd.options.display.max_columns = 999


# # Load Data

# In[ ]:


train = pd.read_csv('../input/shopee-product-detection-student/train.csv')
test = pd.read_csv('../input/shopee-product-detection-student/test.csv')


# In[ ]:


train['category'] = train['category'].astype('object')
category_replace = {0:'00',1:'01',2:'02',3:'03',4:'04',5:'05',6:'06',7:'07',8:'08',9:'09'}
train['category'] = train['category'].replace(category_replace.keys(), category_replace.values()).astype('str')


# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10,4))

train['category'].astype('int').plot.hist(bins=42, ax=ax1);
plt.xlabel('category');
print(train['category'].describe())
print('\nmedian counts: {:.2f}'.format(train['category'].value_counts().median()))
print('mean counts: {:.2f}'.format(train['category'].value_counts().mean()))
ax2.hist(train['category'].value_counts(), bins=42)
ax1.set_xlabel('category'), ax2.set_xlabel('counts')

pd.DataFrame(train['category'].value_counts().sort_values()).T


# The data is left skewed, some of the category has far fewer labels than the majority

# # Visualizing photos

# Let's plot some image

# In[ ]:


PATH = "../input/shopee-product-detection-student/train/train/train/"


# In[ ]:


#we'll plot 10 random photos everytime we run this command
fig, axes = plt.subplots(1,10,figsize=(15,5))
for [filename, category], ax in zip(train.values[np.random.randint(low=1,high=train.shape[0],size=10)], axes):
    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))
    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')


# ## Let's plot the category with the least labels

# In[ ]:


#we'll plot 10 random photos everytime we run this command
fig, axes = plt.subplots(1,10,figsize=(15,5))
for [filename, category], ax in zip(train[train['category'] == '33'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):
    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))
    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')


# Looks like what we need for covid

# In[ ]:


#we'll plot 10 random photos everytime we run this command
fig, axes = plt.subplots(1,10,figsize=(15,5))
for [filename, category], ax in zip(train[train['category'] == '17'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):
    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))
    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')


# In[ ]:


#we'll plot 10 random photos everytime we run this command
fig, axes = plt.subplots(1,10,figsize=(15,5))
for [filename, category], ax in zip(train[train['category'] == '37'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):
    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))
    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')


# ## Let's plot the category with the most labels

# In[ ]:


#we'll plot 10 random photos everytime we run this command
fig, axes = plt.subplots(1,10,figsize=(15,5))
for [filename, category], ax in zip(train[train['category'] == '30'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):
    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))
    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')


# In[ ]:


#we'll plot 10 random photos everytime we run this command
fig, axes = plt.subplots(1,10,figsize=(15,5))
for [filename, category], ax in zip(train[train['category'] == '24'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):
    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))
    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')


# In[ ]:


#we'll plot 10 random photos everytime we run this command
fig, axes = plt.subplots(1,10,figsize=(15,5))
for [filename, category], ax in zip(train[train['category'] == '03'].values[np.random.randint(low=1,high=train[train['category'] == '33'].shape[0],size=10)], axes):
    img = image.load_img(PATH+category+'/'+filename, target_size=(300,300))
    ax.imshow(img), ax.text(0,0,category,c='red'), ax.axis('off')


# It's actually quite easy to guess what are the categories. However, hard coding is both not allowed and not recommended. It'll take forever if we were to manually label every single items

# We are not lucky enough to observe the noisy data Shopee said. We shall leave it to the machine to tell us when we start to build our predictive models.

# In[ ]:




