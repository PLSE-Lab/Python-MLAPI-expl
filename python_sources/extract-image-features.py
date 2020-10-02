#!/usr/bin/env python
# coding: utf-8

# This features are inspired from below kernel

# https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality

# In[ ]:


from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 

from IPython.core.display import HTML 
from IPython.display import Image


# In[ ]:


print('Train')
train = pd.read_csv("../input/train/train.csv")
print(train.shape)

print('Test')
test = pd.read_csv("../input/test/test.csv")
print(test.shape)

print('Breeds')
breeds = pd.read_csv("../input/breed_labels.csv")
print(breeds.shape)

print('Colors')
colors = pd.read_csv("../input/color_labels.csv")
print(colors.shape)

print('States')
states = pd.read_csv("../input/state_labels.csv")
print(states.shape)

target = train['AdoptionSpeed']
train_id = train['PetID']
test_id = test['PetID']
#train.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)
#test.drop(['PetID'], axis=1, inplace=True)


# In[ ]:


import glob

train_image_files = sorted(glob.glob('../input/train_images/*.jpg'))
test_image_files = sorted(glob.glob('../input/test_images/*.jpg'))

print(len(train_image_files), len(test_image_files))
train_image_files[:3], test_image_files[:3]


# In[ ]:


# Images:
train_df_ids = train[['PetID']]
print(train_df_ids.shape)

test_df_ids = test[['PetID']]
print(test_df_ids.shape)


# In[ ]:


train_df_imgs = pd.DataFrame(train_image_files)
train_df_imgs.columns = ['image_filename']
train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])


# In[ ]:


test_df_imgs = pd.DataFrame(test_image_files)
test_df_imgs.columns = ['image_filename']
test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])


# In[ ]:


train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)
print(len(train_imgs_pets.unique()))

test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)
print(len(test_imgs_pets.unique()))


# In[ ]:


train_df_imgs.head()


# In[ ]:


test_df_imgs.head()


# In[ ]:


def getSize(filename):
    #filename = images_path + filename
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    #filename = images_path + filename
    img_size = IMG.open(filename).size
    return img_size 


# In[ ]:


train_df_imgs['image_size'] = train_df_imgs['image_filename'].apply(getSize)
train_df_imgs['temp_size'] = train_df_imgs['image_filename'].apply(getDimensions)
train_df_imgs['width'] = train_df_imgs['temp_size'].apply(lambda x : x[0])
train_df_imgs['height'] = train_df_imgs['temp_size'].apply(lambda x : x[1])
train_df_imgs = train_df_imgs.drop(['temp_size'], axis=1)
train_df_imgs.head()


# In[ ]:


test_df_imgs['image_size'] = test_df_imgs['image_filename'].apply(getSize)
test_df_imgs['temp_size'] = test_df_imgs['image_filename'].apply(getDimensions)
test_df_imgs['width'] = test_df_imgs['temp_size'].apply(lambda x : x[0])
test_df_imgs['height'] = test_df_imgs['temp_size'].apply(lambda x : x[1])
test_df_imgs = test_df_imgs.drop(['temp_size'], axis=1)
test_df_imgs.head()


# In[ ]:


aggs = {
    'image_size': ['min', 'max', 'mean', 'median', "sum"],
    'width': ['min', 'max', 'mean', 'median', "sum"],
    'height': ['min', 'max', 'mean', 'median', "sum"],
}

agg_train_imgs = train_df_imgs.groupby('PetID').agg(aggs)

new_columns = [
    k + '_' + agg for k in aggs.keys() for agg in aggs[k]
]

agg_train_imgs.columns = new_columns

agg_train_imgs = agg_train_imgs.reset_index()
agg_train_imgs.head()


# In[ ]:


agg_test_imgs = test_df_imgs.groupby('PetID').agg(aggs)

new_columns = [
    k + '_' + agg for k in aggs.keys() for agg in aggs[k]
]

agg_test_imgs.columns = new_columns

agg_test_imgs = agg_test_imgs.reset_index()
agg_test_imgs.head()


# In[ ]:


train = train.merge(
    right=agg_train_imgs, how='outer', on='PetID')

print(train.shape)
train.head()


# In[ ]:


test = test.merge(
    right=agg_test_imgs, how='outer', on='PetID')

print(test.shape)
test.head()

