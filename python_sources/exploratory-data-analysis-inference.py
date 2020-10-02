#!/usr/bin/env python
# coding: utf-8

# # Google Landmark Retrieval 2020 - Exploratory Data Analysis
# 
# <img src="https://cdn.getyourguide.com/img/tour_img-2420980-146.jpg" alt="drawing" style="width:780px;"/></br>
# 
# 
# ### Previous Related Competitions
# 
# 1. [Google Landmark Retrieval 2019](https://www.kaggle.com/c/landmark-retrieval-2019)
# 1. [Google Landmark Retrieval Challenge](https://www.kaggle.com/c/landmark-retrieval-challenge)
# 
# References:
# 1. https://www.kaggle.com/seriousran/google-landmark-retrieval-2020-eda
# 2. https://www.kaggle.com/codename007/google-landmark-retrieval-exploratory-analysis
# 3. https://www.kaggle.com/huangxiaoquan/google-landmarks-v2-exploratory-data-analysis-eda

# ## Structure
# 1. Exploratory Data Analysis
#     1. [Training data](#2)
#     2. [Specific basic information](#2)
#     3. [Upload all the data](#4)
#     4. [Test examples](#5)
#     5. [Index examples](#6)
#     6. [Images by_classes](#7)

# In[ ]:


import numpy as np 
import pandas as pd 

import os
import glob
import cv2

import seaborn as sns
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

from scipy import stats

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")


# ## 1.1 Train data
# 
# In this competition, you are asked to develop models that can efficiently retrieve landmark images from a large database. 
# The training set is available in the train/ folder, with corresponding landmark labels in train.csv. 

# In[ ]:


df_train = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
print(f'Train df consumes {df_train.memory_usage().sum() / 1024**2:.4f} MB of RAM and has a shape {df_train.shape}')
df_train.sample(5)


# ### Landmark_id distribuition

# In[ ]:


value_counts = df_train.landmark_id.value_counts().reset_index().rename(columns={'landmark_id': 'count', 'index': 'landmark_id'})
value_counts_sorted = value_counts.sort_values('count')
value_counts


# As we see, the frequency of landmark id 138982 greatly exceeds all the others. Hence, landmark_id equal to 138982 may be either a special number (for instance, referring no non-labeled or mixed images) or just represent the most popular landmark.

# In[ ]:


plt.figure(figsize=(12, 6))
plt.title('landmark_id distribution')
sns.distplot(df_train['landmark_id']);


# ### Training set: number of images per class (dist plot and line plot)
# 
# 

# In[ ]:


plt.figure(figsize=(12,6))
p1=sns.distplot(value_counts, color="b").set_title('Number of images per class')


# In[ ]:


plt.figure(figsize=(12, 6))
sns.set()
plt.title('Training set: number of images per class (line plot logarithmically scaled)')
ax = value_counts['count'].plot(logy=True, grid=True)
locs, labels = plt.xticks()
plt.setp(labels, rotation=30)
ax.set(xlabel="Landmarks", ylabel="Number of images");


# Visualize outliers, min/max or quantiles of the landmarks count

# In[ ]:


sns.set()
ax = value_counts.boxplot(column='landmark_id')
ax.set_yscale('log')


# In[ ]:


sns.set()
res = stats.probplot(df_train['landmark_id'], plot=plt)


# ### Training set: number of images per class(scatter plot)

# In[ ]:


plt.figure(figsize=(12, 6))
sns.set()
landmarks_fold_sorted = value_counts_sorted
ax = landmarks_fold_sorted.plot.scatter(
     x='landmark_id',y='count',
     title='Training set: number of images per class(statter plot)')
locs, labels = plt.xticks()
plt.setp(labels, rotation=30)
ax.set(xlabel="Landmarks", ylabel="Number of images");


# ## 1.2 Specific basic information
# 
# In this section we will derive some interesting and useful facts about the data distribution.

# In[ ]:


threshold = [2, 3, 5, 10, 20, 50, 100]
for num in threshold:    
    print("Number of classes under {}: {}/{} "
          .format(num, (df_train['landmark_id'].value_counts() < num).sum(), 
                  len(df_train['landmark_id'].unique()))
          )


# Visualize top-25 most and least frequent landmark ids

# In[ ]:


sns.set()
plt.figure(figsize=(14, 9))
plt.title('Most frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=value_counts_sorted.tail(25),
            label="Count")
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.show()


# In[ ]:


sns.set()
plt.figure(figsize=(14, 9))
plt.title('Least frequent landmarks')
sns.set_color_codes("pastel")
sns.barplot(x="landmark_id", y="count", data=value_counts_sorted.head(25),
            label="Count")
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.show()


# ## 1.3 Upload all the data
# 
# The query images are listed in the test/ folder, while the "index" images from which you are retrieving are listed in index/. 
# 
# Each image has a unique id. Since there are a large number of images, each image is placed within three subfolders according to the first three characters of the image id (i.e. image abcdef.jpg is placed in a/b/c/abcdef.jpg).
# 
# 0-f in 0-f in 0-f

# In[ ]:


train_list = glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')
test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')
index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')


# In[ ]:


print( 'Query', len(test_list), ' test images in ', len(index_list), 'index images')


# ## 1.4 Test examples

# In[ ]:


def plot_random_images(data_list, seed=2020_07_02, n_rows=3, n_cols=3):
    
    np.random.seed(seed)
    random_ids = np.random.choice(range(len(data_list)), n_rows * n_cols, False)

    plt.rcParams["axes.grid"] = False
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(24, 22))

    curr_row = 0
    for i, random_id in enumerate(random_ids):
        example = cv2.imread(data_list[random_id])
        example = example[:,:,::-1]

        col = i % n_cols
        axarr[col, curr_row].imshow(example)
        if col == n_cols - 1:
            curr_row += 1


# In[ ]:


plot_random_images(test_list)


# ## 1.5 Index examples

# In[ ]:


plot_random_images(index_list)


# ## 1.6 Images by classes

# In[ ]:


all_ids = df_train.landmark_id.unique()

np.random.seed(2020)
n_random, len_row = 5, 3
random_ids = np.append(np.random.choice(all_ids, n_random, False), [138982])


# In[ ]:


plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(n_random+1, len_row, figsize=(len_row*7, 6*(n_random+1)))

curr_row = 0
for random_id in random_ids:
    images = df_train.query(f'landmark_id == {random_id}').sample(len_row)['id']
    for i, img in enumerate(images):
        arg_img = int(np.argwhere(list(map(lambda x: img in x, train_list))).ravel())
        example = cv2.imread(train_list[arg_img])
        example = example[:,:,::-1]

        col = i % len_row
        axarr[curr_row, col].imshow(example)
        if col == len_row - 1:
            curr_row += 1


# ### A few more examples
# (I found the images, depicted using another random seed, representative for other points).

# In[ ]:


np.random.seed(0)
n_random, len_row = 3, 3
random_ids = np.random.choice(all_ids, n_random, False)

plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(n_random, len_row, figsize=(len_row*7, 6*n_random))

curr_row = 0
for random_id in random_ids:
    images = df_train.query(f'landmark_id == {random_id}').sample(len_row)['id']
    for i, img in enumerate(images):
        arg_img = int(np.argwhere(list(map(lambda x: img in x, train_list))).ravel())
        example = cv2.imread(train_list[arg_img])
        example = example[:,:,::-1]

        col = i % len_row
        axarr[curr_row, col].imshow(example)
        if col == len_row - 1:
            curr_row += 1


# ## 1.7 Inference

# ### We can derive several important aspects of the data:
#     1. Some photos are either augmented versions of others (2nd and 3rd images in the last row of the second cell).
#     2. There are a lot of classes containing images of the same places (1st and 5th rows in the first cell, last two rows in the second cell).
#     3. Some classes may contain non-trivial or, what is more probable, noisy data (the last image in row 4 in the first cell).
#     4. The most popular landmark_id refers to photographs with annotation. An interesting fact is the annotation always has the same structure with first number turned upside down. It may probably stand for the photo's id. 
#     5. Index data contains images with people zoomed, while test / train data do not or store less of them.

# In[ ]:




