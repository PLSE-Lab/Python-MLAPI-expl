#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import required libraries
import os
import gc
import sys
import json
import random
from pathlib import Path

import cv2 # CV2 for image manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from imgaug import augmenters as iaa

import seaborn as sns
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold


# In[ ]:


get_ipython().system('pip install tensorflow==1.5')
get_ipython().system('pip install keras==2.1.5')

import tensorflow
print(tensorflow.__version__)
import keras
print(keras.__version__)


# In[ ]:


get_ipython().system('ls /kaggle/input/imaterialist-fashion-2020-fgvc7/')


# In[ ]:


get_ipython().run_cell_magic('time', '', "with open('/kaggle/input/imaterialist-fashion-2020-fgvc7/label_descriptions.json', 'r') as file:\n    label_desc = json.load(file)\nsample_sub_df = pd.read_csv('/kaggle/input/imaterialist-fashion-2020-fgvc7/sample_submission.csv')\ntrain_df = pd.read_csv('/kaggle/input/imaterialist-fashion-2020-fgvc7/train.csv')")


# In[ ]:


train_df.head()


# In[ ]:


sample_sub_df.head()


# In[ ]:


print(f'Shape of training dataset: {train_df.shape}')


# In[ ]:


print(f'# of images in training set: {train_df["ImageId"].nunique()}')
print(f'# of images in test set: {sample_sub_df["ImageId"].nunique()}')


# ### Image size analysis in training dataset

# In[ ]:


pd.DataFrame([train_df['Height'].describe(), train_df['Width'].describe()]).T.loc[['max', 'min', 'mean']]


# ### Height and Width destribution of training images

# In[ ]:


image_shape_df = train_df.groupby("ImageId")["Height", "Width"].first()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ax1.hist(image_shape_df['Height'], bins=100)
ax1.set_title("Height distribution")
ax2.hist(image_shape_df['Width'], bins=100)
ax2.set_title("Width distribution")
plt.show()


# ### Image with minimum height

# In[ ]:


plt.figure(figsize = (70,7))
min_height = list(set(train_df[train_df['Height'] == train_df['Height'].min()]['ImageId']))[0]
plt.imshow(mpimg.imread(f'/kaggle/input/imaterialist-fashion-2020-fgvc7/train/{min_height}.jpg'))
plt.grid(False)
plt.show()


# ### Image with maximum height

# In[ ]:


plt.figure(figsize = (70,7))
max_height = list(set(train_df[train_df['Height'] == train_df['Height'].max()]['ImageId']))[0]
plt.imshow(mpimg.imread(f'/kaggle/input/imaterialist-fashion-2020-fgvc7/train/{max_height}.jpg'))
plt.grid(False)
plt.show()


# ### Image with minimum width

# In[ ]:


plt.figure(figsize = (70,7))
min_width = list(set(train_df[train_df['Width'] == train_df['Width'].min()]['ImageId']))[0]
plt.imshow(mpimg.imread(f'/kaggle/input/imaterialist-fashion-2020-fgvc7/train/{min_width}.jpg'))
plt.grid(False)
plt.show()


# ### Image with maximum width

# In[ ]:


plt.figure(figsize = (70,7))
max_width = list(set(train_df[train_df['Width'] == train_df['Width'].max()]['ImageId']))[0]
plt.imshow(mpimg.imread(f'/kaggle/input/imaterialist-fashion-2020-fgvc7/train/{max_width}.jpg'))
plt.grid(False)
plt.show()


# In[ ]:


area_df = pd.DataFrame()
area_df['ImageId'] = train_df['ImageId']
area_df['area'] = train_df['Height'] * train_df['Width']
min_area = list(set(area_df[area_df['area'] == area_df['area'].min()]['ImageId']))[0]
max_area = list(set(area_df[area_df['area'] == area_df['area'].max()]['ImageId']))[0]


# ### Image with minimum area

# In[ ]:


plt.figure(figsize = (70,7))
plt.imshow(mpimg.imread(f'/kaggle/input/imaterialist-fashion-2020-fgvc7/train/{min_area}.jpg'))
plt.grid(False)
plt.show()


# ### Image with maximum area

# In[ ]:


plt.figure(figsize = (70,7))
plt.imshow(mpimg.imread(f'/kaggle/input/imaterialist-fashion-2020-fgvc7/train/{max_area}.jpg'))
plt.grid(False)
plt.show()


# ## Details about Classes and Attributes

# In[ ]:


num_classes = len(label_desc['categories'])
num_attributes = len(label_desc['attributes'])
print(f'Total # of classes: {num_classes}')
print(f'Total # of attributes: {num_attributes}')


# In[ ]:


categories_df = pd.DataFrame(label_desc['categories'])
attributes_df = pd.DataFrame(label_desc['attributes'])
categories_df


# In[ ]:


pd.set_option('display.max_rows', 300)
attributes_df


# ## Plotting a few training images without any masks

# In[ ]:


def plot_images(size=12, figsize=(12, 12)):
    # First get some images to be plotted
    image_ids = train_df['ImageId'].unique()[:12]
    images=[]
    
    for image in image_ids:
        images.append(mpimg.imread(f'/kaggle/input/imaterialist-fashion-2020-fgvc7/train/{image}.jpg'))
    
    # Plot images in groups of 4 images
    n_groups = 4
    
    count = 0
    for index in range(size // 4):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        for row in ax:
            for col in row:
                col.imshow(images[count])
                col.axis('off')
                count += 1
        plt.show()
    gc.collect()


# In[ ]:


plot_images()


# ## Plotting a few images with given segments

# In[ ]:


def create_mask(size):
    image_ids = train_df['ImageId'].unique()[:size]
    images_meta=[]

    for image_id in image_ids:
        img = mpimg.imread(f'/kaggle/input/imaterialist-fashion-2020-fgvc7/train/{image_id}.jpg')
        images_meta.append({
            'image': img,
            'shape': img.shape,
            'encoded_pixels': train_df[train_df['ImageId'] == image_id]['EncodedPixels'],
            'class_ids':  train_df[train_df['ImageId'] == image_id]['ClassId']
        })

    masks = []
    for image in images_meta:
        shape = image.get('shape')
        encoded_pixels = list(image.get('encoded_pixels'))
        class_ids = list(image.get('class_ids'))
        
        # Initialize numpy array with shape same as image size
        height, width = shape[:2]
        mask = np.zeros((height, width)).reshape(-1)
        
        # Iterate over encoded pixels and create mask
        for segment, (pixel_str, class_id) in enumerate(zip(encoded_pixels, class_ids)):
            splitted_pixels = list(map(int, pixel_str.split()))
            pixel_starts = splitted_pixels[::2]
            run_lengths = splitted_pixels[1::2]
            assert max(pixel_starts) < mask.shape[0]
            for pixel_start, run_length in zip(pixel_starts, run_lengths):
                pixel_start = int(pixel_start) - 1
                run_length = int(run_length)
                mask[pixel_start:pixel_start+run_length] = 255 - class_id * 4
        masks.append(mask.reshape((height, width), order='F'))  # https://stackoverflow.com/questions/45973722/how-does-numpy-reshape-with-order-f-work
    return masks, images_meta


# In[ ]:


def plot_segmented_images(size=12, figsize=(14, 14)):
    # First create masks from given segments
    masks, images_meta = create_mask(size)
    
    # Plot images in groups of 4 images
    n_groups = 4
    
    count = 0
    for index in range(size // 4):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        for row in ax:
            for col in row:
                col.imshow(images_meta[count]['image'])
                col.imshow(masks[count], alpha=0.75)
                col.axis('off')
                count += 1
        plt.show()
    gc.collect()


# In[ ]:


plot_segmented_images()


# ## Analysing Categories and Attributes

# In[ ]:


categories_df = pd.DataFrame(label_desc.get('categories'))
attributes_df = pd.DataFrame(label_desc.get('attributes'))


# In[ ]:


print(f'# of categories: {len(categories_df)}')
print(f'# of attributes: {len(attributes_df)}')


# So there are 46 categories (classes) and 294 attributes. Let's see some of the categories and attributes

# In[ ]:


categories_df.head()


# In[ ]:


attributes_df.head()


# In[ ]:


category_map, attribute_map = {}, {}
for cat in label_desc.get('categories'):
    category_map[cat.get('id')] = cat.get('name')
for attr in label_desc.get('attributes'):
    attribute_map[attr.get('id')] = attr.get('name')


# In[ ]:


train_df['ClassId'] = train_df['ClassId'].map(category_map)
train_df['ClassId'] = train_df['ClassId'].astype('category')


# ### Let's see the class wise distribution of segments in training dataset

# In[ ]:


sns.set(style='darkgrid')
fig, ax = plt.subplots(figsize = (10,10))
sns.countplot(y='ClassId',data=train_df , ax=ax, order = train_df['ClassId'].value_counts().index)
fig.show()


# ### Now let's visualize an image with all its classes and attributes

# In[ ]:


IMAGE_ID = '000b3ec2c6eaffb491a5abb72c2e3e26'


# In[ ]:


# Get the an image id given in the training set for visualization
vis_df = train_df[train_df['ImageId'] == IMAGE_ID]
vis_df['ClassId'] = vis_df['ClassId'].cat.codes
vis_df = vis_df.reset_index(drop=True)
vis_df


# From above table, this image has 8 segmentes and a few attributes. Let's visualize all of them!

# ## Let's first the plot the plain image

# In[ ]:


plt.figure(figsize = (110,11))
image = mpimg.imread(f'/kaggle/input/imaterialist-fashion-2020-fgvc7/train/{IMAGE_ID}.jpg')
plt.grid(False)
plt.imshow(image)
plt.plot()


# In[ ]:


train_df[train_df['ImageId'] == IMAGE_ID]


# ## Now let's plot each segment in a separate image

# In[ ]:


segments = list(vis_df['EncodedPixels'])
class_ids = list(vis_df['ClassId'])
masks = []
for segment, class_id in zip(segments, class_ids):
    
    height = vis_df['Height'][0]
    width = vis_df['Width'][0]
    # Initialize empty mask
    mask = np.zeros((height, width)).reshape(-1)
    
    # Iterate over encoded pixels and create mask
    splitted_pixels = list(map(int, segment.split()))
    pixel_starts = splitted_pixels[::2]
    run_lengths = splitted_pixels[1::2]
    assert max(pixel_starts) < mask.shape[0]
    for pixel_start, run_length in zip(pixel_starts, run_lengths):
        pixel_start = int(pixel_start) - 1
        run_length = int(run_length)
        mask[pixel_start:pixel_start+run_length] = 255 - class_id * 4

    mask = mask.reshape((height, width), order='F')
    masks.append(mask)


# In[ ]:


def plot_individual_segment(*masks, image, figsize=(110, 11)):
    plt.figure(figsize = figsize)
    plt.imshow(image)
    for mask in masks:
        plt.imshow(mask, alpha=0.6)
    plt.axis('off')
    plt.show()


# ## Plotting 1st Segment: ClassId: "Shoe" and no attributes 

# In[ ]:


plot_individual_segment(masks[0], image=image)


# ## Plotting 2nd Segment: ClassId: "shoe"

# In[ ]:


plot_individual_segment(masks[1], image=image)


# ## Plotting 3rd Segment with ClassId: "pants"

# In[ ]:


plot_individual_segment(masks[2], image=image)


# ## Plotting 4th Segment with ClassId: "top, t-shirt, sweatshirt"

# In[ ]:


plot_individual_segment(masks[3], image=image)


# ## Plotting 5th Segment with ClassId: "pocket"

# In[ ]:


plot_individual_segment(masks[4], image=image)


# ## Plotting 6th Segment with ClassId: "sleeve"

# In[ ]:


plot_individual_segment(masks[5], image=image)


# ## Plotting 7th Segment with ClassId: "sleeve"

# In[ ]:


plot_individual_segment(masks[6], image=image)


# ## Plotting 8th segment with Class "neckline"

# In[ ]:


plot_individual_segment(masks[6], image=image)


# Some of the segments have no attributes. Let's check how many such segment exists in training dataset.

# Let's check of missing values in training dataset for columns other than "AttributeIds"

# In[ ]:


train_df[['ImageId', 'EncodedPixels', 'Height', 'Width', 'ClassId']].isna().sum()


# ## Data Preparation and modeling

# In[ ]:


train_df.head()


# In[ ]:


train_df['ClassId'] = train_df['ClassId'].cat.codes


# In[ ]:


train_df

