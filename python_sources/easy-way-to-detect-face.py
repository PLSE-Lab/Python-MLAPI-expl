#!/usr/bin/env python
# coding: utf-8

# ### I just introduce the way to find the face by similarity approach.
# 
# Faces can be detected easily 
# if there are more than 2 images whose sources are the same.

# In[ ]:


import numpy as np
import pandas as pd
import os
import glob
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[ ]:


train_dir = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/"
train_video_files = glob.glob(train_dir+"*.mp4")
train_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T
train_metadata.head()


# In[ ]:


print(train_metadata["original"].value_counts()[0:12])


# In[ ]:


train_frequent=train_metadata[train_metadata["original"]=="qtnjyomzwo.mp4"]

frame_num=0
video_1=list(train_frequent.index)[0]
video_2=list(train_frequent.index)[1]

fig, axes = plt.subplots(1,3, figsize=(30,10))

cap = cv2.VideoCapture(train_dir+video_1)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
_, image = cap.read()
image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cap.release()
axes[0].imshow(image_1[0:500,850:1350,:])
axes[0].title.set_text(f"{video_1}")

cap = cv2.VideoCapture(train_dir+video_2)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
_, image = cap.read()
image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cap.release()
axes[1].imshow(image_2[0:500,850:1350,:])
axes[1].title.set_text(f"{video_2}")

image_3=np.sum((image_1-image_2)**2,axis=2)
axes[2].imshow(image_3[0:500,850:1350])
axes[2].title.set_text("Difference(MSE)")

plt.show()


# First two images are two fake images and the right one is the RMS between them. You can see the large error appears in the face. Face area can be cropped with filter and/or clustering.(I don't show here)
# This can be used to check if your detector is working well.
# 
# The following is other example.

# In[ ]:


train_frequent=train_metadata[train_metadata["original"]=="xngpzquyhs.mp4"]

frame_num=0
video_1=list(train_frequent.index)[0]
video_2=list(train_frequent.index)[1]

fig, axes = plt.subplots(1,3, figsize=(30,10))

cap = cv2.VideoCapture(train_dir+video_1)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
_, image = cap.read()
image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cap.release()
axes[0].imshow(image_1[0:400,800:1250,:])
axes[0].title.set_text(f"{video_1}")

cap = cv2.VideoCapture(train_dir+video_2)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
_, image = cap.read()
image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cap.release()
axes[1].imshow(image_2[0:400,800:1250,:])
axes[1].title.set_text(f"{video_2}")

image_3=np.sum((image_1-image_2)**2,axis=2)
axes[2].imshow(image_3[0:400,800:1250])
axes[2].title.set_text("Difference(MSE>{})")

plt.show()


# I've also checked the similarity among all images.

# In[ ]:


frame_num=0
all_first_image=[]
for i, file_name in tqdm(enumerate(list(train_metadata.index))):
    cap = cv2.VideoCapture(train_dir+file_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    _, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cap.release()
    image=cv2.resize(image,(96,128))
    all_first_image.append(image)
all_first_image=np.array(all_first_image)
thresh=27#if MSE<thresh, regard pixels as the same
similarity_matrix=np.sum(((all_first_image[:,np.newaxis,:,:]-all_first_image[np.newaxis,:,:,:])**2).reshape(400,400,-1)<thresh,axis=2)


# In[ ]:


print(similarity_matrix.shape)
plt.pcolor(similarity_matrix[:20,:20])


# This shows the similarity among first 20 videos.
# For example, you can see video no.3 and no.10 have the same source. (As you know, this is already provided information.)
