#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


data = pd.read_csv('../input/USvideos.csv')


# First we import our US Youtube Video datasets, and we will look what it is;

# In[ ]:


data.info()


# Its first five elements are:

# In[ ]:


data.head()


# While i was handling this data, i can see there are some unique values, and i wont use them while i was showing properties of the data, and correlating  them.

# In[ ]:


data.corr()


# And as we look to correlation matrix we can see there are some strong relations on this dataset. Forexample "likes" and "views", or "dislikes" and "comment_count"
# 
# Now lets it more visulizable correlation:) (For this purpose i will import matplotlib)

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# We can understand that viewers are sharing their opinions if like or dont like  the videos in US Youtube users:)

# Now i will make some plot trials:

# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.likes.plot(kind = 'line', color = 'g',label = 'Likes',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.comment_count.plot(color = 'r',label = 'Comment Counts',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# I just want to show likes and comment counts relations but line plot doest look good, i will try scatter plot now:

# In[ ]:


# Scatter Plot 
# x = likes, y = comment counts
data.plot(kind='scatter', x='likes', y='comment_count',alpha = 0.5,color = 'blue')
plt.xlabel('Likes')              # label = name of label
plt.ylabel('Comment Counts')
plt.title('Like and Comment Counts Scatter Plot')            # title = title of plot


# With same method i wanna look relationship between dislike and comment count

# In[ ]:


# Scatter Plot 
# x = dislikes, y = comment counts
data.plot(kind='scatter', x='dislikes', y='comment_count',alpha = 0.5,color = 'red')
plt.xlabel('Dislikes')              # label = name of label
plt.ylabel('Comment Counts')
plt.title('Dislike and Comment Counts Scatter Plot')            # title = title of plot


# Lets return to data frame and look US youtube players like and dislike videos

# In[ ]:


# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['likes']>500000) & (data['dislikes']>500000)]


# I curios about them and because 3 of two people liked and one of them this like this videos

# 

# and dont wory, i  watched one of them for you:)

# > This was just a look for Youtube dataset, thaks alot. Please share your opinions, So like youtuber bros at the end:** please like and subscribe:)**

# In[ ]:




