#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Importing Libraries**

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Importing Datasets**

# In[5]:


us_videos=pd.read_csv('../input/USvideos.csv')
us_category=pd.read_json('../input/US_category_id.json')


# **Displaying First five data in us_videos dataset**

# In[4]:


us_videos.head()


# **Displaying First five data in us_category dataset**

# In[6]:


us_category.head()


# **Generating descriptive statistics of the data**

# In[7]:


us_videos.describe()


# **Updating the dictionary cdata using update**

# In[8]:


cdata={}
for index, row in us_category.iterrows():
    cid=row['items'].get('id')
    title=row['items'].get('snippet').get('title')
    cdata.update({cid:title})


# In[9]:


def compute_title(i):
    if str(i) in cdata:
        return cdata[str(i)]
    else:
        return np.n


# In[10]:


us_videos['category_title']=us_videos['category_id'].apply(compute_title)


# **Displaying first five data in us_videos dataset**

# In[11]:


us_videos.head(5)


# **The information about the datset**

# In[12]:


us_videos.info()


# **Counting the number of attributes for each instances**

# In[13]:


us_videos[us_videos['description'].isnull()==True].count()


# **A countplot to find how many videos are there in each category**

# In[14]:


plt.figure(figsize=(14,8))
sns.set_style('whitegrid')
sns.countplot(y='category_title',data=us_videos)
plt.ylabel('Category Title')


# **Relationship between views and likes**

# In[15]:


plt.figure(figsize=(14,6))
sns.jointplot(x="views",y="likes",data=us_videos,alpha=0.5)


# **Dropping the description column**

# In[16]:


us_videos.drop('description',axis=1,inplace=True)


# **Relationship between views and comment count**

# In[17]:


plt.figure(figsize=(14,6))
sns.jointplot(x="views",y="comment_count",data=us_videos,alpha=0.5)


# **Finding correlation in data**

# In[18]:


corr_df=us_videos.corr()


# In[19]:


corr_df


# **Heatmap for the correlated data**

# In[20]:


plt.figure(figsize=(14,6))
sns.heatmap(corr_df,annot=True)


# **Relationship between views and comment count in linear regression **

# In[22]:


plt.figure(figsize=(14,6))
sns.jointplot(x='likes',y='comment_count',data=us_videos,color="green",kind='reg')


# **Plotting the graph for likes vs category title.**
# 
# **From the graph we can say that US people like more nonprofits and activism videos**

# In[23]:


plt.figure(figsize=(14,8))
ax=sns.barplot(x='category_title',y='likes',data=us_videos)
plt.xticks(rotation=30)


# **Plotting the graph for dislikes vs category title.**
# 
# **From the graph we can say that US people dislike more nonprofits and activism videos**

# In[24]:


plt.figure(figsize=(14,8))
ax=sns.barplot(x='category_title',y='dislikes',data=us_videos)
plt.xticks(rotation=30)


# **Plotting the graph for comment_count  vs category title.**
# 
# **From the graph we can say that US people comments more on  nonprofits and activism videos**

# In[25]:


plt.figure(figsize=(14,8))
ax=sns.barplot(x='category_title',y='comment_count',data=us_videos)
plt.xticks(rotation=30)


# **From the above plots we conclude the following about US people activity in youtube**
#  There are lot of videos in Entertainment category
#  US peo
# 
