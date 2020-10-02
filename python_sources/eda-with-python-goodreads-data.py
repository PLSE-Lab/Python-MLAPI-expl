#!/usr/bin/env python
# coding: utf-8

# **THIS IS THE FIRST HOMEWORK OF THE DATA SCIENCE AND PYTHON BY DATAI TEAM
# **
# 
# The code here is rather simple as I am a beginner at Python.
# 
# With the Goodread data I will try to explore the book data in Goodreads and visualize the rating, reviews and number of page and find out if there is any correlation between these characteristics. The reason I have choose this dataset is my reading passion. This exercise will also help me to discover new books.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# 1. Lets first load the data and find out about the columns and what type of data it retains

# In[ ]:


df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines = False)
df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.columns


# Found out the num_page column name appears with space so I will use the rename function to edit it.

# In[ ]:


df.rename(columns={'  num_pages':'num_pages'}, inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.nunique()


# In[ ]:


df.duplicated().sum()


# It is good to see there is no duplicated values. So, no need to clean it.

# In[ ]:


df.describe()


# The mean of the Average Book rating is quite high. This means readers tend to give high ratings which makes it hard to differenciate very good books from the rest of the books.

# In[ ]:


df.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(12,12))
sns.heatmap(df.corr(), annot=True, linewidths=0.5, fmt='.2f', ax=ax)


# In[ ]:


df.plot(x='publication_date',y='average_rating', color='r', label='Ratings', linewidth=1, alpha=0.7, grid=True, linestyle=':')
#df.plot(kind='line', color='g', label='Reviews', linewidth=1, alpha=0.7, grid=True, linestyle=':')
plt.xlabel('xaxsis')
plt.ylabel('yaxsis')
plt.title('Ratings vs. Reviews')
plt.show()


# In[ ]:


df.plot(kind='scatter', x='ratings_count',y='text_reviews_count', alpha=.7, color='b')
plt.xlabel('Ratings')
plt.ylabel('Reviews')
plt.title('Scatter')
plt.show()


# In[ ]:


df.average_rating.plot(kind='hist', bins=100, figsize=(8,8))
plt.xlabel('Average Rating')
plt.title('Average Rating Frequency');


# **FILTERING**

# In[ ]:


x = df['num_pages']>2000
df[x]


# In[ ]:


df[np.logical_and(df['average_rating']>4.5, df['ratings_count']>5000)]


# In[ ]:


for index,value in df[['title']][500:505].iterrows():
    print(index," : ",value)


# In[ ]:




