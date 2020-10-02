#!/usr/bin/env python
# coding: utf-8

# # EDA - Google Landmark Retrieval 2020

# According to [Data Description](https://www.kaggle.com/c/landmark-retrieval-2020/data), "In this competition, you are asked to develop models that can efficiently retrieve landmark images from a large database."

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc">
#     <ul class="toc-item">
#         <li><span><a href="#1.-Input-data-overview" data-toc-modified-id="Input-data-overview-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Input data overview</a></span>
#     <ul class="toc-item">
#         <li><span><a href="#1.1.-Number-of-images" data-toc-modified-id="Number-of-images-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Number of images</a></span></li>
#         <li><span><a href="#1.2.-Sample-images" data-toc-modified-id="Sample-images-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Sample images</a></span></li>
#             </ul></li>
#         <li><span><a href="#2.-Training-data" data-toc-modified-id="Training-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Training data</a></span>
#             <ul class="toc-item">
#                 <li><span><a href="#2.1.-landmark_id" data-toc-modified-id="landmark_id-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>landmark_id</a></span></li>
#                 <li><span><a href="#2.2.-Add-image-path-to-train.csv" data-toc-modified-id="Add-image-path-to-train.csv-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Add image path to train.csv</a></span></li>
#                 <li><span><a href="#2.3.-Display-images-per-landmark_id" data-toc-modified-id="Display-images-per-landmark_id-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Display images per landmark_id</a></span></li>
#             </ul></li>
#     </ul>
# </div>

# <hr>
# 
# ## 1. Input data overview
# 
# - The training set is available in the train/ folder, with corresponding landmark labels in train.csv. 
# 
# - The query images are listed in the test/ folder, while the "index" images from which you are retrieving are listed in index/.
# 
# - The provided index/ and test/ images in the publicly available dataset are provided to mock the size and structure of the private data, but are otherwise not directly used.
# 
# - Each image has a unique id. Since there are a large number of images, each image is placed within three subfolders according to the first three characters of the image id (i.e. image abcdef.jpg is placed in a/b/c/abcdef.jpg).
# 
# For more information, please read [Data Description](https://www.kaggle.com/c/landmark-retrieval-2020/data) !

# In[ ]:


import glob
from PIL import Image
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


input_dir = '../input/landmark-retrieval-2020/'
glob.glob(input_dir+'*')


# ### 1.1. Number of images

# In[ ]:


train_images = sorted(glob.glob(input_dir+"train/*/*/*/*.jpg"))
test_images = sorted(glob.glob(input_dir+"test/*/*/*/*.jpg"))
index_images = sorted(glob.glob(input_dir+"index/*/*/*/*.jpg"))

print(f'train : {len(train_images)}         test : {len(test_images)}         index : {len(index_images)}')


# ### 1.2. Sample images

# In[ ]:


def plot_sample_images(images, nrows=2, ncols=4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, int(nrows*2.5)))
    for i, ax in enumerate(axes.flatten()):
        img = plt.imread(images[i])
        ax.imshow(img)
    plt.show()


# In[ ]:


plot_sample_images(train_images)


# In[ ]:


plot_sample_images(test_images)


# In[ ]:


plot_sample_images(index_images)


# ## 2. Training data

# ### 2.1. landmark_id

# In[ ]:


train_df = pd.read_csv(input_dir+'train.csv')
train_df


# Number of landmark_id.

# In[ ]:


train_df.landmark_id.unique().shape


# Counts per landmark_id.

# In[ ]:


train_df.landmark_id.value_counts()


# In[ ]:


fig, axis = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(train_df['landmark_id'], order=train_df['landmark_id'].value_counts().index[:20], ax=axis[0])
sns.countplot(train_df['landmark_id'], order=train_df['landmark_id'].value_counts().index[-20:], ax=axis[1])
axis[0].tick_params(axis='x', labelrotation=90)
axis[1].tick_params(axis='x', labelrotation=90)
axis[1].set(ylim=(0, 10))
plt.show()


# This data is unbalanced.

# ### 2.2. Add image path to train.csv

# In[ ]:


def trans_id_to_path(id):
    path = f'{input_dir}/train/{id[0]}/{id[1]}/{id[2]}/{id}.jpg'
    return path

train_df['path'] = train_df['id'].map(trans_id_to_path)
train_df.head(5)


# In[ ]:


train_df.to_pickle(f'train_path.zip')


# ### 2.3. Display images per landmark_id

# In[ ]:


def display_images_per_id(df, id, n=2):
    nrows = n
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, int(nrows*2.5)))
    cnt = len(df[df['landmark_id']==id])
    print(f'Landmark ID = {id}')
    for i in range(nrows):
        for j in range(ncols):
            img = plt.imread(df[df['landmark_id']==id].path.iloc[random.randint(0, cnt - 1)])
            axes[i, j].imshow(img)
    plt.show()


# In[ ]:


i = random.randint(0, len(train_df)-1)
display_images_per_id(train_df, id=train_df.landmark_id[i])


# In[ ]:


i = random.randint(0, len(train_df)-1)
display_images_per_id(train_df, id=train_df.landmark_id[i])


# In[ ]:


i = random.randint(0, len(train_df)-1)
display_images_per_id(train_df, id=train_df.landmark_id[i])

