#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from matplotlib.patches import Rectangle
from scipy.io import loadmat


# In[ ]:


devkit_path = Path('../input/car_devkit/devkit')
train_path = Path('../input/cars_train/cars_train')
test_path = Path('../input/cars_test/cars_test')


# # DevKit

# In[ ]:


os.listdir(devkit_path)


# ## `README.txt`
# 
# ```
# This file gives documentation for the cars 196 dataset.
# (http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
# 
# ----------------------------------------
# Metadata/Annotations
# ----------------------------------------
# Descriptions of the files are as follows:
# 
# -cars_meta.mat:
#   Contains a cell array of class names, one for each class.
# 
# -cars_train_annos.mat:
#   Contains the variable 'annotations', which is a struct array of length
#   num_images and where each element has the fields:
#     bbox_x1: Min x-value of the bounding box, in pixels
#     bbox_x2: Max x-value of the bounding box, in pixels
#     bbox_y1: Min y-value of the bounding box, in pixels
#     bbox_y2: Max y-value of the bounding box, in pixels
#     class: Integral id of the class the image belongs to.
#     fname: Filename of the image within the folder of images.
# 
# -cars_test_annos.mat:
#   Same format as 'cars_train_annos.mat', except the class is not provided.
# ```
# 
# From the `README.txt` file, we have the three meta data filles oppened bellow

# In[ ]:


cars_meta = loadmat(devkit_path/'cars_meta.mat')
cars_train_annos = loadmat(devkit_path/'cars_train_annos.mat')
cars_test_annos = loadmat(devkit_path/'cars_test_annos.mat')


# ## Loading Labels

# In[ ]:


labels = [c for c in cars_meta['class_names'][0]]
labels = pd.DataFrame(labels, columns=['labels'])
labels.head(10)


# ## Loading Cars Train

# In[ ]:


frame = [[i.flat[0] for i in line] for line in cars_train_annos['annotations'][0]]
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
df_train = pd.DataFrame(frame, columns=columns)
df_train['class'] = df_train['class']-1 # Python indexing starts on zero.
df_train['fname'] = [train_path/f for f in df_train['fname']] #  Appending Path
df_train.head()


# ### Merging labels

# In[ ]:


df_train = df_train.merge(labels, left_on='class', right_index=True)
df_train = df_train.sort_index()
df_train.head()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.info()


# ## Loading Cars Test

# In[ ]:


frame = [[i.flat[0] for i in line] for line in cars_test_annos['annotations'][0]]
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
df_test = pd.DataFrame(frame, columns=columns)
df_test['fname'] = [test_path/f for f in df_test['fname']] #  Appending Path
df_test.head()


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_test.info()


# ## Displaying Image

# In[ ]:


# Returns (Image, title, rectangle patch) for drawing
def get_assets(df, i):
    is_train = df is df_train
    folder = train_path if is_train else test_path
    image = Image.open(df['fname'][i])
    title = df['labels'][i] if is_train else 'Unclassified'

    xy = df['bbox_x1'][i], df['bbox_y1'][i]
    width = df['bbox_x2'][i] - df['bbox_x1'][i]
    height = df['bbox_y2'][i] - df['bbox_y1'][i]
    rect = Rectangle(xy, width, height, fill=False, color='r', linewidth=2)
    
    return (image, title, rect)


# In[ ]:


def display_image(df, i):
    image, title, rect = get_assets(df, i)
    print(title)

    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.gca().add_patch(rect)


# In[ ]:


display_image(df_train, 0)


# ## Displaying Range Image

# In[ ]:


def display_range(end, start = 0):

    n = end - start
    fig, ax = plt.subplots(n, 2, figsize=(15, 5*end))

    for i in range(start, end):
        line = i - start
        
        im, title, rect = get_assets(df_train, i)
        sub = ax[line, 0]
        sub.imshow(im)
        sub.axis('off')
        sub.set_title(title)
        sub.add_patch(rect)
        
        im, title, rect = get_assets(df_test, i)
        sub = ax[line, 1]
        sub.imshow(im)
        sub.axis('off')
        sub.set_title(title)
        sub.add_patch(rect)
        
    plt.show()


# In[ ]:


display_range( 10,0 )


# ## Cars' Distribution

# In[ ]:


freq_labels = df_train.groupby('labels').count()[['class']]
freq_labels = freq_labels.rename(columns={'class': 'count'})
freq_labels = freq_labels.sort_values(by='count', ascending=False)
freq_labels.head()


# In[ ]:


freq_labels.head(50).plot.bar(figsize=(15,10))
plt.xticks(rotation=90);
plt.xlabel("Cars");
plt.ylabel("Count");


# In[ ]:




