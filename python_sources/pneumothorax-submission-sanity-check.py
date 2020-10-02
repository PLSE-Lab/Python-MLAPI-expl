#!/usr/bin/env python
# coding: utf-8

# # Submission Sanity Check

# In[ ]:


# Imports
import numpy as np
import pandas as pd
import random

from glob import glob
import pydicom

from PIL import Image

from matplotlib import pyplot as plt
import matplotlib

import os
print(os.listdir("../input"))


# I am trying to build a model for the Pneumothorax Segmentation competition and to be honest I am not doing well. That is why I am trying to look at my submission files to get some insights of how I could improve the model.
# 
# You can also use the code below to analyze your submission or just to have fun.

# ## Load data

# Load the csv with submission:

# In[ ]:


# Load submission file 
submission = pd.read_csv("../input/pneumothorax-submission/submission09.csv")
submission = submission.set_index('ImageId')

submission.head()


# Load images from the test dataset:

# In[ ]:


# Load test data
datafilepath = '../input/siim-train-test/siim/'
fns = sorted(glob(datafilepath + 'dicom-images-test/*/*/*.dcm'))


# ## Plot single images

# Following code helps to plot a single image and predicted mask:

# In[ ]:


# mask functions:
def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(runStart);
                    rle.append(runLength);
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    if lastColor == 255:
        rle.append(runStart)
        rle.append(runLength)

    return " ".join(str(rle))

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


# In[ ]:


# Function to plot single image with mask
def get_image_and_pred(idx, fns, submission):
    '''
    Function to plot single image with mask
    INPUT:
        idx - number of the image to plot
        fns - test data files
        submission - submission dataframe
        
    OUTPUT:
        has_pneumo - (True or False) whether the image has pneumothorax (according to the prediction)
        image - PIL image for the image from the test set
        rle_mask - rle mask of the mask predicted for thr image
        mask - PIL image for the predicted mask
    '''
    fname = fns[idx]
    
    dataset = pydicom.read_file(fname)
    np_image = np.expand_dims(dataset.pixel_array, axis=2)
    
    image = Image.fromarray(np_image.reshape(1024, 1024) , 'L')
    
    if '-1' in submission.loc[fname.split('/')[-1][:-4],'EncodedPixels']:
        np_mask = np.zeros((1024, 1024, 1), dtype=np.bool)
        rle_mask = None
        
        has_pneumo = False    
    else:
        rle_mask = rle2mask(submission.loc[fname.split('/')[-1][:-4],'EncodedPixels'], 1024, 1024) 
        
        has_pneumo = True
    
    np_mask = np.transpose(rle_mask)
    mask = Image.fromarray(np_mask)
    
    return has_pneumo, image, rle_mask, mask


# In[ ]:


# get image and prediction
pneumo, image, rle_mask, mask = get_image_and_pred(0, fns, submission)


# In[ ]:


# Function to plot the prediction
def plot_xray(image, rle_mask):
    '''
    Function to plot a single prediction:
    INPUT:
        image - PIL image from the test dataset
        rle_mask -  mask predicted for the image
    '''
    fig, axs = plt.subplots(1, 2, figsize=(20,10))

    #plot the original data
    axs[0].imshow(image, cmap='bone') 
    axs[0].axis('off')
    axs[0].set_title('Without mask')

    #plot image and add the mask
    axs[1].imshow(image, cmap='bone')
    axs[1].axis('off')  
    axs[1].imshow(np.transpose(rle_mask), alpha=0.3, cmap="Reds")    
    axs[1].set_title('With mask')

    # set suptitle
    plt.suptitle('Images with pneumothorax')
    plt.show()


# Try to plot single predictions:

# In[ ]:


# try plotting the prediction
plot_xray(image, rle_mask)


# This doesn't look so bad:
# * Pneumothorax is a small area as it usually is.
# * The model found pneumothorax near the edge of the lung, where I would look for it.

# In[ ]:


# plot some more predictions
pneumo, image, rle_mask, mask = get_image_and_pred(7, fns, submission)
plot_xray(image, rle_mask)


# Hmmm, It looks that model found something that looked like pneumothorax!

# ## Plot image grid

# Now I would like to plot a grid of images with predicted masks:

# In[ ]:


images_pneumo = submission[submission['EncodedPixels'] != '-1']


# In[ ]:


# Plot several images with predicted pneumothorax
def plot_xray_grid(fns, submission):
    '''
    Function to plot several predictions
    INPUT:
        fns - files from the test dataset
        submission - submission file
    '''
    fig, axs = plt.subplots(2, 5, figsize=(20,8))
    idx = 0
    n = 0
    
    while n < 5:
        
        try:
            
            # get image and prediction
            pneumo, image, rle_mask, mask = get_image_and_pred(idx, fns, submission)

            if pneumo:

                #plot the original data
                axs[0, n].imshow(image, cmap='bone') 
                axs[0, n].axis('off')
                axs[0, n].set_title('Without mask')

                #plot image and add the mask
                axs[1, n].imshow(image, cmap='bone')
                axs[1, n].axis('off')  
                axs[1, n].imshow(np.transpose(rle_mask), alpha=0.3, cmap="Reds")    
                axs[1, n].set_title('With mask')

                n = n + 1
            
        except:
            pass
            
        idx = idx + 1
        
    # set suptitle
    plt.suptitle('Images with pneumothorax and masks', fontsize = 16)
    plt.show()


# In[ ]:


plot_xray_grid(fns, images_pneumo)


# Well, the model may have some point :)
# I have to really thank [@agentili](https://www.kaggle.com/agentili), who pointed out the errors in my code. The model wasn't so bad after all. My other code was.
