#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h1><bold><center>EDA - Google Landmark Retrieval 2020</center></bold></h1>
# 
# ![800006316768a007.jpg](attachment:800006316768a007.jpg)
# <br>
# <br>
# <br>
# <br>
# 
# #### What is the DELF(DEep Local Feature) module?
# The pre-trained DELF(DEep Local Feature) can be used for image retrieval as a drop-in replacement for other keypoint detectors and descriptors. It describes each noteworthy point in a given image with 100-dimensional vectors known as feature descriptor.
# 
# The image below shows the DELF correspondences of two images.
# 
# ![horseshoe-delf.jpg](attachment:horseshoe-delf.jpg)
# 
# <br>
# <br>
# <br>
# <br>
# 
# The DELF Image retrieval system can be decomposed into four main blocks:
# 
# * Dense localized feature extraction,
# * Keypoint selection,
# * Dimensionality reduction,
# * Indexing and retrieval.
# 
# 
# 
# 

# # Data Overview
# 
# 
# ### 1. Imports

# In[ ]:


import pandas as pd
from bokeh.plotting import figure, output_notebook, show
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

get_ipython().run_line_magic('matplotlib', 'inline')
output_notebook()


# In[ ]:


df_train =  pd.read_csv('../input/landmark-retrieval-2020/train.csv')


# In[ ]:


df_train.head(10)


# In[ ]:


df_train.shape


# In[ ]:


y = df_train['landmark_id'].unique()


# In[ ]:


x = df_train.landmark_id.value_counts()


# In[ ]:


max(x)


# In[ ]:


y = df_train.id[df_train['landmark_id'][x]]
y[:1]


# ## Image Distribution With Respect to Landmark ID

# In[ ]:


plot = figure(plot_width = 3050, plot_height = 500)
plot.vbar(x=x.index, top=x, width = 0.75, color = "steelblue")


show(plot)


# In[ ]:


sns.set()
landmarks_fold_sorted = pd.DataFrame(df_train['landmark_id'].value_counts())
landmarks_fold_sorted.reset_index(inplace=True)
landmarks_fold_sorted.columns = ['landmark_id','count']
landmarks_fold_sorted = landmarks_fold_sorted.sort_values('landmark_id')
ax = landmarks_fold_sorted.plot.scatter(     x='landmark_id',y='count',
     title='Training set: number of images per class(statter plot)')
locs, labels = plt.xticks()
plt.setp(labels, rotation=30)
ax.set(xlabel="Landmarks", ylabel="Number of images")


# ## Examples from Train Image Dataset

# In[ ]:


f = plt.figure(figsize=(15,15))

axes = []
rows = 3
cols = 3
for i in range(cols*rows):
    a = df_train['id'][i]
    image_path = '../input/landmark-retrieval-2020/train/'+a[0]+'/'+a[1]+'/'+a[2]+'/'+a+'.jpg'
    image1 = cv2.imread(image_path)
    image1 = image1[:,:,::-1]
    axes.append( f.add_subplot(rows, cols, i+1))
    subplot_title=('Landmark ID - '+str(df_train['landmark_id'][i]))
    axes[-1].set_title(subplot_title)  
    plt.imshow(image1)
f.tight_layout()


# In[ ]:





# In[ ]:




