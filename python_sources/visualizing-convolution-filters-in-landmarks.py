#!/usr/bin/env python
# coding: utf-8

# # Google Landmark Retrieval 2020 Convolutional Filter Visualization

# <img src="https://gitcdn.xyz/cdn/Tony607/blog_statics/da023c6fa09cc1c6de8ce8ab9358f74f21f56579/images/landmark/eiffel-tower.jpg" alt="drawing" style="width:780px;"/></br>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import glob
import cv2 
from math import ceil

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# **References:**
# Idea based on [tarun paparaju](https://www.kaggle.com/tarunpaparaju)'s [plant pathology's kernel](https://www.kaggle.com/tarunpaparaju/plant-pathology-2020-eda-models)
# 
# Some EDA codes based off [Akim Tsivgun](https://www.kaggle.com/maschasap)'s [EDA kernel](https://www.kaggle.com/maschasap/exploratory-data-analysis-inference)

# ## Train Data
# 
# In this competition, you are asked to develop models that can efficiently retrieve landmark images from a large database. The training set is available in the train/ folder, with corresponding landmark labels in train.csv.

# In[ ]:


train_df = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
train_df.head()


# In[ ]:


train_list = glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')
test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')
index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')
len(train_list)
len(test_list)
len(index_list)


# In[ ]:


all_ids = train_df.landmark_id.unique()
len(all_ids)
all_ids


# ## Visualization

# **Helper Function**

# In[ ]:


def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=ceil(len(img_matrix_list) / ncols), ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()


# We will just take 2 unique `landmark_id` as examples to visualize.

# In[ ]:


num_of_samples = 3

# train_df.loc[train_df['id']=='a00009431492c304']
# we will be using this landmark and its corresponding id 14112 for visualizations

landmark_14112 = train_df.query(f'landmark_id == {14112}').sample(num_of_samples)['id']
landmark_14112


# In[ ]:


landmark_14112_image_list = []
title_list = [1,2,3]
for i, img in enumerate(landmark_14112):
    arg_img = int(np.argwhere(list(map(lambda x: img in x, train_list))).ravel())
    landmark_14112_img = cv2.imread(train_list[arg_img])[:,:,::-1]
    landmark_14112_image_list.append(landmark_14112_img)

plot_multiple_img(landmark_14112_image_list, title_list, ncols=3, main_title="")


# ## Convolution <a id="2.3"></a>
# 
# Convolution is a rather simple algorithm which involves a kernel (a 2D matrix) which moves over the entire image, calculating dot products with each window along the way. The GIF below demonstrates convolution in action.
# 
# <center><img src="https://i.imgur.com/wYUaqR3.gif" width="450px"></center>
# 
# The above process can be summarized with an equation, where *f* is the image and *h* is the kernel. The dimensions of *f* are *(m, n)* and the kernel is a square matrix with dimensions smaller than *f*:
# 
# <center><img src="https://i.imgur.com/9scTOGv.png" width="350px"></center>
# <br>
# 
# In the above equation, the kernel *h* is moving across the length and breadth of the image. The dot product of *h* with a sub-matrix or window of matrix *f* is taken at each step, hence the double summation (rows and columns). 
# 
# Let me show you how visualizing a convolutional filter is cool!

# In[ ]:


def conv_horizontal(img):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    kernel = np.ones((3,3), np.float32)
    kernel[1] = np.array([0,0,0],np.float32)
    kernel[2] = np.array([-1,-1,-1],np.float32)
    conv = cv2.filter2D(img, -1, kernel)
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(conv)
    ax[1].set_title('Convolved Image', fontsize=24)
    plt.show()

def conv_vertical(img):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    kernel = np.ones((3,3), np.float32)
    kernel[0] = np.array([1,0,-1])
    kernel[1] = np.array([1,0,-1])
    kernel[2] = np.array([1,0,-1])
    conv = cv2.filter2D(img, -1, kernel)
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(conv)
    ax[1].set_title('Convolved Image', fontsize=24)
    plt.show()


# The below visualizes the convolutional filter that detects horizontal edges.

# In[ ]:


conv_horizontal(landmark_14112_image_list[0])
conv_horizontal(landmark_14112_image_list[1])
conv_horizontal(landmark_14112_image_list[2])


# The below visualizes the convolutional filter that detects vertical edges.

# In[ ]:


conv_vertical(landmark_14112_image_list[0])
conv_vertical(landmark_14112_image_list[1])
conv_vertical(landmark_14112_image_list[2])

