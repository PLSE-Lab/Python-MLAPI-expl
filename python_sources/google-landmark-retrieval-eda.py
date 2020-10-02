#!/usr/bin/env python
# coding: utf-8

# ## Let's GO!!!
# 
# In this Kernel We will be doing EDA on Google Landmark Retrieval:
# - [Understanding the data](#1)
# - [Plotting the data](#2)
# - [Displaying the images](#3)
# 
# <p><font size='4' color='green'> If you like this kernel then please consider giving an upvote !</font></p>

# In[ ]:


from IPython.display import Image
Image("../input/imagefile/place.jpg")


# ### Understanding the data <a id="1" ></a>

# In[ ]:


#Importing necessary libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob


# In[ ]:


import os
print(os.listdir("../input/landmark-retrieval-2020"))


# In[ ]:


#reading the training and test data
train_data = pd.read_csv('../input/landmark-retrieval-2020/train.csv')

print("Training data size:",train_data.shape)


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


train_data.head()
train_data['landmark_id'][33]


# In[ ]:


#Displaying number of unique URLs & ids
len(train_data['landmark_id'].unique())


# In[ ]:


len(train_data['id'].unique())


# ### Plotting the data <a id="2" ></a>

# In[ ]:


plt.title('Distribution')
sns.distplot(train_data['landmark_id'])


# In[ ]:


sns.set()
print(train_data.nunique())
train_data['landmark_id'].value_counts().hist()


# In[ ]:


from scipy import stats
sns.set()
res = stats.probplot(train_data['landmark_id'], plot=plt)


# ### Displaying the images <a id="3" ></a>

# In[ ]:


test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')
index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')
train_list= glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')


# In[ ]:


plt.rcParams["axes.grid"] = True
f, axarr = plt.subplots(6, 5, figsize=(24, 22))

curr_row = 0
for i in range(30):
    example = cv2.imread(test_list[i])
    example = example[:,:,::-1]
    
    col = i%6
    axarr[col, curr_row].imshow(example)
    if col == 5:
        curr_row += 1

