#!/usr/bin/env python
# coding: utf-8

# skript to compare real and fake face images from the first frame of the train videos
# with inspiration from https://www.kaggle.com/robikscube/kaggle-deepfake-detection-introduction
# thanks rob mulla!

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import cv2
from matplotlib import pyplot as plt

train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
sample_submission = pd.read_csv("../input/deepfake-detection-challenge/sample_submission.csv")


# In[ ]:


# download face_recognition
get_ipython().system(' pip install face_recognition')
import face_recognition as fr


# In[ ]:


# find all fake videos
metadata = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json').T
print(metadata.head())
print(len(metadata))

list_fake_videos = list(metadata.loc[metadata['label']=='FAKE',:].index.values)
print(len(list_fake_videos))


# In[ ]:


#prepare plotting of ten videos and their first frame
fig, axes = plt.subplots(10, 2, figsize=(15, 40))
axes = np.array(axes)
axes = axes.reshape(-1)
ax_ix = 0
ax_max = 20

# choose from which videos to display the frames
ind0 = 0
ind1 = 100
padding = 0

for vid in list_fake_videos[ind0:ind1]:
    orig_vid = metadata.loc[vid,'original']
    # check if original video exists in the directory, many do not exist
    if not(os.path.isfile(train_dir + orig_vid) ):
        #print(f'could not find real {orig_vid} for {vid}:')
        continue
    # image from fake video
    cap = cv2.VideoCapture(train_dir + vid)
    success, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cap.release()
    face_locations = fr.face_locations(image)
    if len(face_locations) == 0:
        print(f'Could not find face in {vid} FAKE')
        continue
    
    top, right, bottom, left = face_locations[0] #first face only
    image = image[top-padding:bottom+padding, left-padding:right+padding]
    
    # image from corresponding real video
    cap = cv2.VideoCapture(train_dir + orig_vid)
    success, orig_image = cap.read()
    if not(success):
        print(f'could capture in {orig_vid} for {vid}:')
        continue
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    cap.release() 
    face_locations = fr.face_locations(orig_image)
    if len(face_locations) == 0:
        print(f'Could not find face in {orig_vid}')
        continue
    top, right, bottom, left = face_locations[0] #first face only
    orig_image = orig_image[top-padding:bottom+padding, left-padding:right+padding]
    # plot
    
    axes[ax_ix].imshow(image)
    axes[ax_ix].xaxis.set_visible(False)
    axes[ax_ix].yaxis.set_visible(False)
    axes[ax_ix].set_title(f'{vid} FAKE')
    ax_ix = ax_ix +1
    
    axes[ax_ix].imshow(orig_image)
    axes[ax_ix].xaxis.set_visible(False)
    axes[ax_ix].yaxis.set_visible(False)
    axes[ax_ix].set_title(f'{orig_vid} REAL')
    ax_ix = ax_ix +1
    if ax_ix >=ax_max:
        break

plt.grid(False)
plt.show()    


# In[ ]:





# In[ ]:




