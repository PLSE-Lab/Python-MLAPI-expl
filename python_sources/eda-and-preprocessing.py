#!/usr/bin/env python
# coding: utf-8

# ## Summary
# This kernel does preliminary exploratory data analysis on this dataset and explores some of the pre-processing techniques, some of which are proven to be potentially useful. Some code are borrowed or adapted from other public kernels and discussion threads. Credits are listed below.
# 
# #### Thanks to:
# @Neuron Engineer for his excellent EDA code https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping#data
# 
# @Bharat Singh for his great starter kernel https://www.kaggle.com/bharatsingh213/keras-resnet-tta
# 
# People in this discussion https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/102613#latest-614367

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import cv2
import albumentations
import matplotlib.pyplot as plt
import seaborn as sns
import random

print(tf.__version__)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Define some important global variables here

# In[ ]:


CLASSS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
SEED = 77
random.seed(SEED)
IMG_CHANNELS = 3
IMG_WIDTH = 512

# These are used for histogram equalization
clipLimit=2.0 
tileGridSize=(8, 8)  

channels = {"R":0, "G": 1, "B":2}


# ## Load data and check distribution

# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/sample_submission.csv")
print(sample_submission.head())
test_file = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")
train_file = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
print(test_file.head())
print(train_file.head())


# In[ ]:


# Now check the distribution of train images
print(len(train_file))
train_file['diagnosis'].hist(figsize = (8,4))


# We can see that this dataset is quite imbalanced.
# Next we will visualize some of the training images

# ## Display Original Images

# In[ ]:


def display_samples(df, columns=4, rows=3):
    fig=plt.figure(figsize=(4*columns, 3*rows))
    
    random_indices = random.sample(range(0, len(train_file)), columns*rows)
    count = 0
    for i in random_indices:
        image_path = df.loc[i,'id_code']
        image_rating = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig.add_subplot(rows, columns, count+1)
        count += 1
        plt.title(image_rating)
        plt.imshow(img)
    
    plt.tight_layout()

display_samples(train_file, 4, 8)


# ### Some meta-data features of these images
# * They are actually very large
# * They can be in very different lighting conditions ==> Consider this in augmentations
# * They vary largely in size
# * They have quite different proportion of dark fringes

# In[ ]:


sample_img_path = random.choice(train_file["id_code"])
sample_img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{sample_img_path}.png')
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
print(sample_img.shape)
plt.imshow(sample_img)


# Now try to gain some visual intuition into healthy and unhealthy patients' pictures.
# 
# Following code borrowed from @Bharat Singh

# In[ ]:


def draw_img(imgs, class_label='0'):
    fig, axis = plt.subplots(2, 6, figsize=(15, 6))
    for idnx, (idx, row) in enumerate(imgs.iterrows()):
        imgPath = (f'../input/aptos2019-blindness-detection/train_images/{row["id_code"]}.png')
        img = cv2.imread(imgPath)
        row = idnx // 6
        col = idnx % 6
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axis[row, col].imshow(img)
    plt.suptitle(class_label)
    plt.show()


# In[ ]:


CLASS_ID = 0
draw_img(train_file[train_file.diagnosis == CLASS_ID].head(12), CLASSS[CLASS_ID])


# In[ ]:


CLASS_ID = 4
draw_img(train_file[train_file.diagnosis == CLASS_ID].head(12), CLASSS[CLASS_ID])


# > ## Basic Image Pre-Processing

# **Now we see the three different channels and see if one channel may contain most of the useful information** 
# 
# The idea of single channel (green) images and the histogram equalization comes from this topic: https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/102613#latest-614367
# 
# The CLAHE (Contrast Limited Adaptive Histogram Equalization) used here
# https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
# 
# 
# 

# In[ ]:


# HE --> Histogram Equalization: True to apply CLAHE to the color channel image

print(channels)
print(clipLimit)
print(tileGridSize)

def display_single_channel_samples(df, columns=4, rows=3, channel = "G", HE = False):
    fig=plt.figure(figsize=(4*columns, 3*rows))
    random.seed(SEED) # This lines make sure that all the following function calls will
                    # show the same set of randomly selected images
    random_indices = random.sample(range(0, len(train_file)), columns*rows)
    
    count = 0
    for i in random_indices:
        # Load images and convert to RGB
        image_path = df.loc[i,'id_code']
        image_rating = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply some pre-processing
        img = img[:,:,channels[channel]]
        if HE: #If the histogram equalization is applied
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            img = clahe.apply(img) #This is for creating the image with a higher contrast
        else:
            pass
        
        # Actually drawing stuff 
        fig.add_subplot(rows, columns, count+1)
#         fig.add_subplot()
        count += 1
        plt.title(image_rating)
        plt.imshow(img)
    
    plt.tight_layout()


# **Green Channel WITHOUT histogram equalization**

# In[ ]:


display_single_channel_samples(train_file, 4, 3)


# ** Green Channel WITH Histogram Equalization**
# 
# We can see that this is **slightly better** than the un-equalized version of these images as the contrast has been increased. In the green channel, the vessels are much more clear.
# 
# Next, we try to visualiza another two channels, each **WITH** HE applied.
# 
# Now, we can see that 

# In[ ]:


display_single_channel_samples(train_file,4,3, "G", HE = True)


# ** Red Channel WITH histogram equalization**

# In[ ]:


display_single_channel_samples(train_file,4,3, "R", HE = True)


# ** Blue Channel WITH histogram equalization**

# In[ ]:


display_single_channel_samples(train_file,4,2, "B", HE = False)


# In[ ]:


display_single_channel_samples(train_file,4,2, "B", HE = True)


# From the comparison above, we can see that histogram equalization is indeed helping out the blue channel **a little bit**. But it's still not providing much high-quality information.

# #### Conclusion:
# The green channel is indeed very helpful. The contrast can be effectively adjusted by a simple line of code. I will probably try to use pure green channel to fit a model, as an experiment.

# ### Resize & Crop & Combine
# As described in @Neuron engineer's EDA kernel, another good way to pre-process the RGB images is Ben Grahem's method (https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition), which is simple yet powerful. The following function implements this method.

# In[ ]:


def resize_bens(df, columns=4, rows=3, sigmaX = 20, img_width = IMG_WIDTH): # Assume image is square 
    fig=plt.figure(figsize=(4*columns, 3*rows))
    
    random.seed(SEED)
    random_indices = random.sample(range(0, len(train_file)), columns*rows)
    count = 0
    for i in random_indices:
        image_path = df.loc[i,'id_code']
        image_rating = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width, img_width))
        img = cv2.addWeighted ( img,4, cv2.GaussianBlur(img , (0,0) , sigmaX) ,-4 ,128)
        
        fig.add_subplot(rows, columns, count+1)
        count += 1
        plt.title(image_rating)
        plt.imshow(img)
    
    plt.tight_layout()


# In[ ]:


resize_bens(train_file, 4, 4)


# #### Try some different **sigmaX** values.
# We can see below that maybe sigmaX = 50 is too high, causing some bright white fringes of picture at location (0,0).
# 
# Next we tried sigmaX = 16. It looks very similar to sigmaX = 10

# In[ ]:


resize_bens(train_file,4,4, sigmaX = 50)


# In[ ]:


resize_bens(train_file,4,4, sigmaX = 16)


# ### Crop them!
# From the examples above, it's intuitive to not set the sigmaX any where too high.
# 
# Next, we want to be able to crop these images so that the dark fringes have approx. the same distribution among all images fed into the network. The code below are directly adapted from the kernel of @Neuron Engineer

# In[ ]:


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


# In[ ]:


def resize_bens_and_crop(df, columns=4, rows=3, sigmaX = 20, img_width = IMG_WIDTH): # Assume image is square 
    fig=plt.figure(figsize=(4*columns, 3*rows))
    
    random.seed(SEED)
    random_indices = random.sample(range(0, len(train_file)), columns*rows)
    count = 0
    for i in random_indices:
        image_path = df.loc[i,'id_code']
        image_rating = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # First crop, then resize.
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (img_width, img_width))
        
        # Applying Ben's method
        img = cv2.addWeighted ( img,4, cv2.GaussianBlur(img , (0,0) , sigmaX) ,-4 ,128)
        
        fig.add_subplot(rows, columns, count+1)
        count += 1
        plt.title(image_rating)
        plt.imshow(img)
    
    plt.tight_layout()


# In[ ]:


resize_bens_and_crop(train_file, 4, 4, sigmaX = 10)


# ## Finally
# Here we provide an intuitive comparison between the effectiveness of using SINGLE BLUE CHANNEL WITH HISTOGRAM EQUALIZATION and using RGB IMAGE WITH BEN'S METHOD

# ** Don't forget to add cropping and resizing functions to the SINGLE CHANNEL method we used before **

# In[ ]:


# HE --> Histogram Equalization: Try to apply CLAHE to the color channel image

print(channels)
print(clipLimit)
print(tileGridSize)

def display_single_channel_crop_resize(df, columns=4, rows=3, channel = "G", HE = False):
    fig=plt.figure(figsize=(4*columns, 3*rows))
    random.seed(SEED) # This lines make sure that all the following function calls will
                    # show the same set of randomly selected images
    random_indices = random.sample(range(0, len(train_file)), columns*rows)
    
    count = 0
    for i in random_indices:
        # Load images and convert to RGB
        image_path = df.loc[i,'id_code']
        image_rating = df.loc[i,'diagnosis']
        img = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_path}.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop and then resize the image
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (IMG_WIDTH, IMG_WIDTH))
        
        # Apply some pre-processing
        img = img[:,:,channels[channel]]
        if HE: #If the histogram equalization is applied
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            img = clahe.apply(img) #This is for creating the image with a higher contrast
        else:
            pass
        
        # Actually drawing stuff 
        fig.add_subplot(rows, columns, count+1)
#         fig.add_subplot()
        count += 1
        plt.title(image_rating)
        plt.imshow(img)
    
    plt.tight_layout()


# In[ ]:


resize_bens_and_crop(train_file, 5, 7, sigmaX = 10)


# In[ ]:


display_single_channel_crop_resize(train_file, 5, 7, HE = True)


# In[ ]:




