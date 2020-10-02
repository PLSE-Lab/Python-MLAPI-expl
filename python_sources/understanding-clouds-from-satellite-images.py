#!/usr/bin/env python
# coding: utf-8

# # About Competition
# - **Objective**: build a model to classify cloud organization patterns from satellite images.
# - **Evaluation**: This competition is evaluated on the mean [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient).
# - The leaderboard score is the mean of the Dice coefficients for each <Image, Label> pair in the test set.
# - The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.
# - While the images in Train and Test are 1400 x 2100 pixels, the predictions should be scaled down to a 350 x 525 pixel image.
# - **Submission** 
#     - file should be in csv format, with a header and columns names : **Image_Label, EncodedPixels**.
#     - Each row represents a single predicted cloud type segmentation for the given Image, and predicted Label
#     - Should have the the same number of rows as **num_images * num_labels**.
#     - Encode segment for cloud type in an image into a single row, even if there are several non-contiguous cloud type locations.
#     - If there is no area of a certain cloud type for an image, the corresponding EncodedPixels prediction should be left blank.
# - For each image in the test set, segment the regions of each cloud formation label (Fish, Flower, Gravel, Sugar). Each image has at least one cloud formation, and can possibly contain up to all all four.
# - **Data**
#     - An image might be stitched together from two orbits. The remaining area, which has not been covered by two succeeding orbits, is marked black.
#     - **train.csv** contains the run length encoded segmentations for each image-label pair in the train_images
# - Task is to predict the segmentations masks of each of the 4 cloud types (labels) for each image. (Your prediction masks should be scaled down to 350 x 525 px).
# 
# # Notes
# - Shallow clouds play a huge role in determining the Earth's climate.

# In[ ]:


import numpy as np
import pandas as pd
import os, random


# In[ ]:


get_ipython().system(" ls '/kaggle/input/understanding_cloud_organization'")


# In[ ]:


Data_dir = "/kaggle/input/understanding_cloud_organization/"

print("There are {} Training images".format(len(os.listdir(Data_dir + 'train_images'))))
print("There are {} Testing images".format(len(os.listdir(Data_dir + 'test_images'))))


# In[ ]:


train_df = pd.read_csv(Data_dir + "train.csv")
train_df.head()


# In[ ]:


train_df[['Image', 'Label']] = train_df['Image_Label'].str.split('_', expand=True)
train_df.head()


# In[ ]:





# In[ ]:


# There are 4 Lables for each image, but some have EncodedPixels and some don't
train_df.groupby("Image").count().head()


# In[ ]:


train_df.isnull().sum()
# If a label is present then it's encodedPixels is given, if not it's NaN


# In[ ]:




