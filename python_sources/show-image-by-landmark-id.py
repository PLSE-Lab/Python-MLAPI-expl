#!/usr/bin/env python
# coding: utf-8

# I made a [notebook](https://www.kaggle.com/shumpeiabe/show-image-and-landmark-id) that shows a image when you give a row number.
# 
# But in this competition, I thought displaying image from landmark_id is more useful than it. So, this is it.

# In[ ]:


import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_dir = '/kaggle/input/landmark-retrieval-2020/train/'
train = pd.read_csv('/kaggle/input/landmark-retrieval-2020/train.csv')


# In[ ]:


def get_image_path(df, num, train_dir):
    folder = []
    for i in range(3):
        folder.append(df['id'][num][i])
        
    _path = train_dir + '{}/{}/{}/'.format(folder[0], folder[1], folder[2]) 
    file_name = df['id'][num] + '.jpg'
    image_path = _path + file_name
    return image_path


# In[ ]:


def get_img_from_landmark_id(df, landmark_id, train_dir):
    landmark_df = df[df['landmark_id']==landmark_id]
    
    if len(landmark_df)==0:
        return "no picture of landmark_id={}".format(landmark_id)
    image_path = []
    for i in range(len(landmark_df)):
        image_path.append(get_image_path(df, i, train_dir))
    
    print("pictures of landmark_id={}".format(landmark_id))
    plt.figure(figsize=(8.0, 20.0))
    plt.subplots_adjust(wspace=0.6, hspace=1)
    for i, file_name in enumerate(image_path):
        display_image = plt.imread(file_name)
        plt.title(landmark_df['id'][i])
        plt.subplot(len(image_path),2,i+1)
        plt.imshow(display_image)


# In[ ]:


get_img_from_landmark_id(train, 1, train_dir)


# In[ ]:




