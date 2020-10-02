#!/usr/bin/env python
# coding: utf-8

# Started on 2 July 2019
# 
# **References:**
# 1. https://www.kaggle.com/whizzkid/crop-images-using-bounding-box
# 2. https://www.kaggle.com/guillaumedesforges/loading-the-cropped-dogs-seamlessly-with-pytorch
# 3. https://www.kaggle.com/guillaumedesforges/usable-complete-data-loading-utility
# 4. https://towardsdatascience.com/processing-xml-in-python-elementtree-c8992941efd2

# # Introduction

# In[ ]:


import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))


# #### The files are:
# * all-dogs.zip - All dog images contained in the [Stanford Dogs Dataset][1]
# * Annotations.zip - Class labels, bounding boxes
# 
# #### The dataset is from the [Stanford Dogs Dataset][1]. It is useful to refer to the [webpage][1]. 
# [1]: http://vision.stanford.edu/aditya86/ImageNetDogs/

# #### What I learned from scanning the [Stanford Dogs Dataset][1]:
# * There are 20,580 images. These are in the 'all-dogs' directory. The filename of the .jpg is the identifier.
# * There are 120 sub-folders in the 'Annotation' directory. Each sub-folder represents a dog breed, and there are ~150 to 200 annotation files in each folder and they correspond to the images in the 'all-dogs' directory via the same identifier filename.
# * The annotation files contain the dog breed labels and bounding boxes.
# * There are images with people in them. The bounding boxes in the respective annotation files define where the dogs are in the images.
# * There are also images with several dogs. For these, there will be more than one bounding box in the corresponding annotation file.
# [1]: http://vision.stanford.edu/aditya86/ImageNetDogs/

# # Load & crop images to bounding boxes

# In[ ]:


import glob
image = glob.glob('../input/all-dogs/all-dogs/*')
breed = glob.glob('../input/annotation/Annotation/*')
annot = glob.glob('../input/annotation/Annotation/*/*')
print(len(image), len(breed), len(annot))


# __Note:__ The number of images in the 'all-dogs' folder is 20579, which is one less than that in the 'Annotation' folder. Hmm..

# In[ ]:


# Let's take a look at the content of an annotation file. I choose one with two dogs in the image, and
# there are two bounding boxes specified. 
get_ipython().system('cat ../input/annotation/Annotation/n02097658-silky_terrier/n02097658_98')


# #### Function to extract bounding box values from annotation files.
# Python has a built-in library called ElementTree that has functions to read and manipulate XML files.

# In[ ]:


import xml.etree.ElementTree as ET


# In[ ]:


def get_bbox(annot):
    """
    This extracts and returns values of bounding boxes
    """
    xml = annot
    tree = ET.parse(xml)
    root = tree.getroot()
    objects = root.findall('object')
    bbox = []
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox.append((xmin,ymin,xmax,ymax))
    return bbox


# In[ ]:


# test
bbox = get_bbox('../input/annotation/Annotation/n02097658-silky_terrier/n02097658_98')
print(bbox[0], bbox[1], len(bbox))


# In[ ]:


def get_image(annot):
    """
    Retrieve the corresponding image given annotation file
    """
    img_path = '../input/all-dogs/all-dogs/'
    file = annot.split('/')
    img_filename = img_path+file[-1]+'.jpg'
    return img_filename


# #### Let's see some images

# In[ ]:


plt.figure(figsize=(10,10))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.axis("off")
    dog = get_image(annot[i])
    im = Image.open(dog)
    im = im.resize((64,64), Image.ANTIALIAS)
    plt.imshow(im)


# #### Let's compare the above when cropped to bounding boxes

# In[ ]:


plt.figure(figsize=(10,10))
for i in range(12):
    bbox = get_bbox(annot[i])
    dog = get_image(annot[i])
    im = Image.open(dog)
    for j in range(len(bbox)):
        im = im.crop(bbox[j])
        im = im.resize((64,64), Image.ANTIALIAS)
    plt.subplot(3,4,i+1)
    plt.axis("off")
    plt.imshow(im)


# #### Let's check the image with two dogs

# In[ ]:


test = '../input/annotation/Annotation/n02097658-silky_terrier/n02097658_98'
box = get_bbox(test)
dog = get_image(test)
im = Image.open(dog)
plt.imshow(im)


# In[ ]:


plt.figure(figsize=(6,6))
for j in range(len(box)):
    im = Image.open(dog)
    im = im.crop(box[j])
    im = im.resize((64,64), Image.ANTIALIAS)
    plt.subplot(1,2, j+1)
    plt.axis("off")
    plt.imshow(im)


# 
