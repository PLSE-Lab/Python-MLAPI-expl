#!/usr/bin/env python
# coding: utf-8

# # 2018 Data Science Bowl
# 
# ## Objective : Find the nuclei in divergent images to advance in medical discovery

# ### Importing necessary libraries and datasets

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import cv2
import math
import seaborn as sns
import json

sns.set()
sns.set_palette("husl")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

RANDOM_SEED=75

OUTPUT_PATH = './'
CONTOUR_EXTRACT_MODE = cv2.RETR_TREE

# Any results you write to the current directory are saved as output.


# In[2]:


# Ensuring that only the folders are picked up and any junk files in the same path are ignored

train_ids = [x for x in os.listdir(TRAIN_PATH) if os.path.isdir(TRAIN_PATH+x)]
test_ids = [x for x in os.listdir(TEST_PATH) if os.path.isdir(TEST_PATH+x)]


# In[3]:


# Create a pandas dataframe combining all images and marking them as train or test. 
# It is a way we can do a comparison across all images.

df = pd.DataFrame({'id':train_ids,'train_or_test':'train'})
df = df.append(pd.DataFrame({'id':test_ids,'train_or_test':'test'}))

df.groupby(['train_or_test']).count()


# There are 670 training images and 65 test images, which makes the test-set less than 10% of the training set, and this sounds OK for our analysis purpose.

# ### Building the paths for the individual images

# In[4]:


df['path'] = df.apply(lambda x:'../input/stage1_{}/{}/images/{}.png'.format(x[1],x[0],x[0]), axis=1)


# In[5]:


from sklearn.cluster import KMeans

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


# In[6]:


def get_image_info(path, clusters=2):
    image = cv2.imread(path)
    height,width,_ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters = clusters)
    clt.fit(image)
    hist = centroid_histogram(clt)
    
    bg_idx, fg_idx = 0, clusters-1
    if hist[bg_idx] < hist[fg_idx]:
        bg_idx, fg_idx = clusters-1, 0
    
    bg_red, bg_green, bg_blue = clt.cluster_centers_[bg_idx]
    fg_red, fg_green, fg_blue = clt.cluster_centers_[fg_idx]
    
    bg_color = sum(clt.cluster_centers_[bg_idx])/3
    fg_color = sum(clt.cluster_centers_[fg_idx])/3
    max_color_pct = hist[bg_idx]
    min_color_pct = hist[fg_idx]
    
    return (pd.Series([height,width,
                       bg_red, bg_green, bg_blue, bg_color,
                       fg_red, fg_green, fg_blue, fg_color,
                       hist[bg_idx],hist[fg_idx],
                       fg_color < bg_color]))


# In[ ]:


image_info = os.path.join(OUTPUT_PATH,'images.json')

if os.path.isfile(image_info):
    with open(image_info, 'r') as datafile:
        data = json.load(datafile)
        df = pd.read_json(path_or_buf=data, orient='records')
        data = None
else:
    names = ['height','width',
             'bg_red', 'bg_green', 'bg_blue','bg_color',
             'fg_red', 'fg_green', 'fg_blue','fg_color',
             'bg_color_pct','fg_color_pct','invert']

    df[names] = df['path'].apply(lambda x: get_image_info(x))
    df['shape'] = df[['height','width']].apply(lambda x: '{:04d}x{:04d}'.format(x[0], x[1]), axis=1)

    with open(image_info, 'w') as outfile:
        json.dump(df.to_json(orient='records'), outfile)


# In[ ]:


len(df['shape'].unique()),len(df['width'].unique()), len(df['height'].unique())


# # Image Distribution
# 
# Now, we will have a look how the data is distributed/spread across all the images.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

agg = df[['shape','train_or_test','id']].groupby(['shape','train_or_test']).count().unstack()
agg.columns = agg.columns.droplevel()

agg.plot.barh(stacked=True,figsize=(15,5))
plt.show()


# In[ ]:


agg[agg['train'].isnull()]

