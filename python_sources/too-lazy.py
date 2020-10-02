#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

get_ipython().system('ls ../input/understanding_cloud_organization/')


# In[ ]:


seed = 1234
np.random.seed(seed)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_path = Path("../input/understanding_cloud_organization/")
train = pd.read_csv(data_path / "train.csv")
sub = pd.read_csv(data_path / "sample_submission.csv")

print("Number of training samples: ", len(train))
print("Number of test samples: ", len(sub))


# In[ ]:


def rle_decode(mask_rle, shape=(1400, 2100)):
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction


# In[ ]:


train.head()


# In[ ]:


# Let's clear up some mess
train_images_path = "../input/understanding_cloud_organization/train_images/"
vals = train["Image_Label"].str.split("_", expand=True)
train["image"] = vals[0]
train["label"] = vals[1]
train["image_path"] = train_images_path + train["image"]
train.head()


# In[ ]:


# How many null values are there?
print("Number of null values in the data")
train.isnull().sum()


# That is very strange. Out of `22K` samples, `~11K` don't have encoded pixel values? Interesting!

# In[ ]:


# Unique labels
train["label"].unique()


# In[ ]:


# Distribution of labels
train["label"].value_counts().plot(kind="bar", figsize=(15, 5))
plt.show()


# Maybe I am not sleeping very well. Let me know if you find anything wrong in this analysis. I am unable to digest the fact that all labels have same number of samples 

# In[ ]:


# Drop the null values for now
train_clean = train.dropna().reset_index(drop=True)
train_clean.head()


# In[ ]:


sample_indices = []
to_select = 4
for label in train_clean['label'].unique():
    label_indices = np.random.choice(train_clean.index[train_clean["label"]==label], size=to_select)
    sample_indices += label_indices.tolist()


# In[ ]:


from skimage.io import imread


# In[ ]:


f,ax = plt.subplots(4,4, figsize=(20,10))
for i, idx in enumerate(sample_indices):
    img = imread(train_clean.iloc[idx]["image_path"])
    mask_rle = train_clean.iloc[idx]["EncodedPixels"]
    mask = rle_decode(mask_rle)
    label = train_clean.iloc[idx]["label"]
    
    ax[i//4, i%4].imshow(img)
    ax[i//4, i%4].imshow(mask, alpha=0.5, cmap='gray')
    ax[i//4, i%4].set_title(label)
    ax[i//4, i%4].axis('off')

plt.show()


# In[ ]:




