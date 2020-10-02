#!/usr/bin/env python
# coding: utf-8

# # Beginner's Guide to Image Augmentation & Transforms
# 

# Hello! If you are a beginner and learning to handle image data, then Image Augmentation & Transforms is often confusing. Some may wonder why it is needed and what is the use.
# 
# What are the changes that occur and why is it necessary. We discuss these in brief in the notebook
# 
# I will be using the Dog Breed database, as we are intuitively more familiar with dogs than human protiens :)
# 
# We use both Python as well as PyTorch in this notebook!
# 
# Hope this helps...

# # Import Libraries

# In the start always import the libraries that you feel you may use or need. Over time, build a list of libraries that you use and use it in all the notebooks you are working. You will naturally strat using some of the libraries that you are comfortable with and this will help.

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import cv2
import random
from random import randint
import time


import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from scipy import ndimage

import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder

from tqdm.notebook import tqdm

from sklearn.metrics import f1_score


# # Read the data

# In[ ]:


DATA_DIR = '../input/dog-breed-identification'


TRAIN_DIR = DATA_DIR + '/train'                           
TEST_DIR = DATA_DIR + '/test'                             

TRAIN_CSV = DATA_DIR + '/labels.csv'                     
TEST_CSV = DATA_DIR + '/submission.csv' 


# # Add more Data Fields

# I will not dwelve into modeling here and focus on image transfors
# 
# As a result I will look at only the train file
# 
# I will read and add some data fields that I find useful

# In[ ]:


data_df = pd.read_csv(TRAIN_CSV)
data_df.head(10)


# What do you observe? a list of image names and breed. Lets add more details to the dataframe
# 
# We create a dictionary of all the breeds

# In[ ]:


labels_names=data_df["breed"].unique()
labels_sorted=labels_names.sort()

labels = dict(zip(range(len(labels_names)),labels_names))
labels 


# I like to use numbers instead of names for labels. Lets add the numbers as labels to the dataframe

# In[ ]:



lbl=[]
for i in range(len(data_df["breed"])):
    temp=list(labels.values()).index(data_df.breed[i])
    lbl.append(temp)

    
data_df['lbl'] = lbl
#data_df['lbl'] = data_df['lbl'].astype(str)
data_df.head()


# Lets also add the path of each image to the file. 

# In[ ]:


path_img=[]
for i in range(len(data_df["id"])):
    temp=TRAIN_DIR + "/" + str(data_df.id[i]) + ".jpg"
    path_img.append(temp)

data_df['path_img'] =path_img
data_df.head()


# Any other field you would like to add? Please make a note in comments. Thanks!

# # Exlporatory Data Analysis (EDA)

# Let us look at the data and make some initial conclusions

# In[ ]:


num_images = len(data_df["id"])
print('Number of images in Training file:', num_images)
no_labels=len(labels_names)
print('Number of dog breeds in Training file:', no_labels)


# Ok! we have over 10,000 images for 120 dog breeds.
# 
# Are images equally distributed between all dog breeds?
# 
# Let's plot a graph and see!

# In[ ]:


bar = data_df["breed"].value_counts(ascending=True).plot.barh(figsize = (30,120))
plt.title("Distribution of the Dog Breeds", fontsize = 20)
bar.tick_params(labelsize=16)
plt.show()


# In[ ]:


data_df["breed"].value_counts(ascending=False)


# We observe that the distribution is not equal. Scottish deerhound has 126 images
# while eskimo dog and briard breeds have 66 images

# # Image Analysis

# Let us display 20 picture of the dataset with their labels

# In[ ]:


fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 15),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(data_df.path_img[i]))
    ax.set_title(data_df.breed[i])
plt.tight_layout()
plt.show()


# What do you observe?
# 
# All images are of differnt sizes
# 
# The backgrounsd vary- some have humans, and other items in the backgrounds
# 
# Also some images are not vertical - e.g., the lakeland terrier in the lower night

# # Image Transforms using Python 

# Let us work on some image transforms using Python
# and later we will use Pytorch to do the same
# 
# Lets start with resizing images

# In[ ]:


random_img=randint(0,len(data_df.path_img))
img_path=data_df.path_img[random_img]
img= plt.imread(img_path)

plt.imshow(img)
plt.title("Original image")
plt.show()

plt.imshow(cv2.resize(img, (150,150)))
plt.title("After resizing")
plt.show()


# Lets try to rotate the images...

# In[ ]:


random_img=randint(0,len(data_df.path_img))
img_path=data_df.path_img[random_img]
img= plt.imread(img_path)

plt.imshow(img)
plt.title("Original image")
plt.show()


#rotation angle in degree

rotated1 = ndimage.rotate(img, 90)
plt.imshow(rotated1)
plt.title("Image rotated 90 degrees")
plt.show()


# As you can observe the originalt height of the image is retained while the width changes
# this creates an issue of differnt sizes and lenghts of images
# 
# Let us do both resize and rotation
# 

# In[ ]:


random_img=randint(0,len(data_df.path_img))
img_path=data_df.path_img[random_img]
img= plt.imread(img_path)

plt.imshow(img)
plt.title("Original image")
plt.show()


img=cv2.resize(img, (150,150))
turn =90

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(ndimage.rotate(img, i*90))
    ax.set_title("After resizing rotated "+ str(i*90) +" degrees")
plt.tight_layout()
plt.show()


# What are the benefits?
# Deep Learning and Neural Networks need a lot of images
# by rotating images we are adding multiple extra images from one image
# 
# Note that I have rotated by 90 degrees  but one may rotate by any random degree that that wants to rotate the image
# 
# Similarly, by blocking parts of images, cropping (removing part of images) and adding jitters, we can both augument images (add to the number of images) and transform them to make it more helpful for the neural network to classify

# # Image transforms using PyTorch

# Now that we have a understanding of the transforms using Python, lets do the same using PyTorch

# In[ ]:


#imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = T.Compose([
#this will resize the image 
    T.Resize(256),   
   
#Randomly change the brightness, contrast and saturation of an image
#    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),    

#this will remove parts (crop) the Image at a random location.   
#    T.RandomCrop(32, padding=4, padding_mode='reflect'),   

#Horizontally flip (rotate by 180 degree) the given image randomly; default is 50% of images
    T.RandomHorizontalFlip(), 
    
#Rotate the image by angle -here by 10%
    T.RandomRotation(10),
    
#convert it to a tensor   
    T.ToTensor()

#Normalize a tensor image with mean and standard deviation - here with the Imagenet stats
#    T.Normalize(*imagenet_stats,inplace=True), 
    
#Randomly selects a rectangle region in an image and erases its pixels.    
#    T.RandomErasing(inplace=True)
])


# Here I focus only on the train transform. Please make sure you make the same transforms in the validation set as well
# 
# Note some of the commands are not running as they are as coments due to the # symbol
# remove the # symbol and see how the images below change. 
# 
# This will help you visualise the impact of each transform
# 
# [Read more about transforms here (click here)](https://pytorch.org/docs/stable/torchvision/transforms.html)

# In[ ]:


class DogDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id, img_label = row['id'], row['lbl']
        img_fname = self.root_dir + "/" + str(img_id) + ".jpg"
        img = Image.open(img_fname)
        if self.transform:
            img = self.transform(img)
        return img, img_label


# In[ ]:


data_ds = DogDataset(data_df, TRAIN_DIR, transform=train_tfms)


# In[ ]:


def show_sample(img, target, invert=True):
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute(1, 2, 0))
    print('Labels:', labels[target])


# # View Sample Images after Transform

# In[ ]:


show_sample(*data_ds[241])


# In[ ]:


show_sample(*data_ds[149])


# Do you notice that the transforms are most likely differnt for both images?
# This is because the transforms are added randomly. In most liklihood each image will have some differnt rotation and.or flip and other transforms

# ***Try this out!***
# 
# and try with using differnt transforms from the link and by removing the # in the code
# 
# **Please share your comments and feedback.**

# # Thank you! Hope you like it!
