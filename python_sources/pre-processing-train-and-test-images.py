#!/usr/bin/env python
# coding: utf-8

# # Updates
# 
# 28/07/2019: Added circle_crop_v2. Here we resize the image after the first crop before drawing the circle. The result is that a larger portion of the zoomed in images is retained, though it will be somewhat stretched. I think a slightly stretch is better than losing so much information.

# # Summary
# 
# Based on findings in this kernel:
# 
# https://www.kaggle.com/taindow/be-careful-what-you-train-on
# 
# I wanted to ensure that train and test images were preprocessed effectively so that the network was not able to learn the idiosyncrasies of how different target groups had been pre-processed before the competition. Image processing is not something I'm super familiar with, so suggestions on improving the code to run more effeciently are much appreciated.
# 
# # References
# 
# Cropping black areas: https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping

# # Function

# In[ ]:


import os
import glob
import cv2
import numpy as np

def crop_image_from_gray(img,tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """  
    
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img


def circle_crop(img):   
    """
    Create circular crop around image centre    
    """    
    
    img = cv2.imread(img)
    img = crop_image_from_gray(img)    
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    
    return img 

def circle_crop_v2(img):
    """
    Create circular crop around image centre
    """
    img = cv2.imread(img)
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img


# # Image Types
# 
# The basic idea is that I want to ensure all images are presented to the network in the right way. In generally I think there are 4 different "types" of images in the dataset.
# 
# 1. Spaceboy: rectangular image, no cropping
# 2. Cropboy: rectangular, lossy vertical cropping
# 3. Squareboy: square image, tight cropping
# 4. Supercropboy: rectangular, lossy vertical and horizontal cropping
# 
# Interestingly, most of the test data is 4, while most of the training data is 1 or 3. Some examples are below.

# In[ ]:


from matplotlib import pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4))

spaceboy = cv2.imread('../input/train_images/1df0a4c23c95.png')
ax[0].imshow(cv2.cvtColor(spaceboy, cv2.COLOR_BGR2RGB))
ax[0].axis('off')

cropboy = cv2.imread('../input/train_images/0a1076183736.png')
ax[1].imshow(cv2.cvtColor(cropboy, cv2.COLOR_BGR2RGB))
ax[1].axis('off')

squareboy = cv2.imread('../input/train_images/0e3572b5884a.png')
ax[2].imshow(cv2.cvtColor(squareboy, cv2.COLOR_BGR2RGB))
ax[2].axis('off')

supercropboy = cv2.imread('../input/train_images/698d6e422a80.png')
ax[3].imshow(cv2.cvtColor(supercropboy, cv2.COLOR_BGR2RGB))
ax[3].axis('off')


# Applying the janky function from above:

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4))

spaceboy = circle_crop('../input/train_images/1df0a4c23c95.png')
ax[0].imshow(cv2.cvtColor(spaceboy, cv2.COLOR_BGR2RGB))
ax[0].axis('off')

cropboy = circle_crop('../input/train_images/0a1076183736.png')
ax[1].imshow(cv2.cvtColor(cropboy, cv2.COLOR_BGR2RGB))
ax[1].axis('off')

squareboy = circle_crop('../input/train_images/0e3572b5884a.png')
ax[2].imshow(cv2.cvtColor(squareboy, cv2.COLOR_BGR2RGB))
ax[2].axis('off')

supercropboy = circle_crop('../input/train_images/698d6e422a80.png')
ax[3].imshow(cv2.cvtColor(supercropboy, cv2.COLOR_BGR2RGB))
ax[3].axis('off')


# Not sure if information loss is worth it, but will experiment.

# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4))

spaceboy = circle_crop_v2('../input/train_images/1df0a4c23c95.png')
ax[0].imshow(cv2.cvtColor(spaceboy, cv2.COLOR_BGR2RGB))
ax[0].axis('off')

cropboy = circle_crop_v2('../input/train_images/0a1076183736.png')
ax[1].imshow(cv2.cvtColor(cropboy, cv2.COLOR_BGR2RGB))
ax[1].axis('off')

squareboy = circle_crop_v2('../input/train_images/0e3572b5884a.png')
ax[2].imshow(cv2.cvtColor(squareboy, cv2.COLOR_BGR2RGB))
ax[2].axis('off')

supercropboy = circle_crop_v2('../input/train_images/698d6e422a80.png')
ax[3].imshow(cv2.cvtColor(supercropboy, cv2.COLOR_BGR2RGB))
ax[3].axis('off')

