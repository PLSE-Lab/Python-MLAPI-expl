#!/usr/bin/env python
# coding: utf-8

# This is the **1st** kernel I share with you.
# In this notebook we like to check the **correlation** between different sort of features. This enables us to see if there are any similarity or contrast beetween different attributes.

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("../input/USvideos.csv")
data.head()


# We're not going to use textuary features; Therefore we remove them.

# In[4]:


drop_list = ['video_id', 'trending_date', 'title', 'category_id', 'publish_time', 'tags', 'thumbnail_link', 'comments_disabled', 'ratings_disabled', 'video_error_or_removed', 'description']
tempData = data.drop(drop_list, axis=1)


# In[5]:


tempData.head()


# In[6]:


f,ax = plt.subplots(figsize=(9, 7))
sns.heatmap(tempData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# <h2>Observations</h2>
# * Views and Likes are highly correlated.
# * If people didn't like something, there probably happened a long argument via comments.

# I know. This is not enough, but it's good for start. Moreover, you can learn how to draw heatmaps using sns, how to remove features and so on.

# Alright! Let's go do some sorting stuffs.
# Let's sort the data-farme, and do it for likes.

# In[8]:


tempData = tempData.sort_values(['likes']).reset_index(drop=True)
tempData.head()


# As you can see there are many channels with lots of views and no likes nor dislikes.
# What do we do with them?
# I mean if we try to build a predictive model, do we have to consider them as noisy; Therefore remove them?

# In[ ]:




