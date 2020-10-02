#!/usr/bin/env python
# coding: utf-8

# Many thanks to Jasper and her kernel hier: https://www.kaggle.com/jesperdramsch/intro-to-dicom-chest-xrays-data-viz-and-unets
# Taking a look at her kernel is worth the while. This kernel is based on Jasper's kernel with the goal to visualize all 10 images.
# I was wandering if un untrained eye can see the Pneumothorax on these sample images. I can honestly say, I can't see anything. Can you? Please leave a comment :) 

# In[ ]:


import os
import glob

import pydicom

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import patches as patches

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '../input')
from mask_functions import rle2mask


# In[ ]:


ls '../input/sample images/'


# In[ ]:


IMAGE_PATH = '../input/sample images/'
IMAGE_MEDIA_TYPE = '.dcm'
IMAGE_SIZE = 1024


# In[ ]:


train_rle_sample = pd.read_csv(IMAGE_PATH + 'train-rle-sample.csv', header=None, index_col=0)
train_rle_sample.head(10).reset_index()


# The first five images are clean, the last five images have pneumothorax.

# In[ ]:


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def show_dcm_info(dataset, image_name):
    print("Image............:", image_name)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)

def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()
    
def plot_with_mask_and_bbox(dataset, mask_encoded, figsize=(20,10)):
    mask_decoded = rle2mask(mask_encoded, 1024, 1024).T
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))
    rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
    patch = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
    ax[0].imshow(dataset.pixel_array, cmap=plt.cm.bone)
    ax[0].imshow(mask_decoded, alpha=0.3, cmap="Reds")
    ax[0].add_patch(patch)
    ax[0].set_title('With Mask')

    patch = patches.Rectangle((cmin,rmin),cmax-cmin,rmax-rmin,linewidth=1,edgecolor='r',facecolor='none')
    ax[1].imshow(dataset.pixel_array, cmap=plt.cm.bone)
    ax[1].add_patch(patch)
    ax[1].set_title('Without Mask')
    plt.show()

def show_image(image_name):
    dataset = pydicom.dcmread(IMAGE_PATH + image_name + IMAGE_MEDIA_TYPE)
    show_dcm_info(dataset, image_name)
    
    mask_encoded = train_rle_sample.loc[image_name].values[0]
    if mask_encoded == '-1':    
        plot_pixel_array(dataset)
    else:
        plot_with_mask_and_bbox(dataset, mask_encoded)


# Let's see the first five images - without pneumothorax.

# In[ ]:


show_image('1.2.276.0.7230010.3.1.4.8323329.1000.1517875165.878027')


# In[ ]:


show_image('1.2.276.0.7230010.3.1.4.8323329.10000.1517875220.938530')


# In[ ]:


show_image('1.2.276.0.7230010.3.1.4.8323329.10001.1517875220.930580')


# In[ ]:


show_image('1.2.276.0.7230010.3.1.4.8323329.10002.1517875220.939397')


# In[ ]:


show_image('1.2.276.0.7230010.3.1.4.8323329.10003.1517875220.942420')


# Not let's take a look at the images with Pneumothorax...

# In[ ]:


show_image('1.2.276.0.7230010.3.1.4.8323329.4904.1517875185.355709')


# In[ ]:


show_image('1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290')


# In[ ]:


show_image('1.2.276.0.7230010.3.1.4.8323329.4440.1517875182.865105')


# In[ ]:


show_image('1.2.276.0.7230010.3.1.4.8323329.4982.1517875185.837576')


# In[ ]:


show_image('1.2.276.0.7230010.3.1.4.8323329.12743.1517875241.599591')


# In[ ]:





# In[ ]:




