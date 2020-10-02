#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data.head(5)


# In[ ]:


data.columns


# In[ ]:


# line plot
data.popularity.plot(kind='line', color='g', label='popularity',grid='True', linestyle=':')
data.revenue.plot(kind='line', color='r', label='revenue', linewidth=1, grid='True', alpha=0.5, linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('line Plot')
plt.show()


# In[ ]:


# Scatter Plot
# x = budget, y = revenue
data.plot(kind='scatter', x='budget', y='revenue', alpha=0.5, color='brown')
plt.xlabel('budget')
plt.ylabel('revenue')
plt.title('Attack budget revenue Scatter Plot')
plt.show()


# In[ ]:


# Histogram
data.vote_average.plot(kind='hist', bins= 50, figsize=(12,12))
plt.show()


# In[ ]:


data.vote_average.plot(kind='hist', bins=50)
plt.clf()


# In[ ]:


series = data['budget']        # data['Defense'] = series
print(type(series))
data_frame = data[['budget']]  # data[['Defense']] = data frame
print(type(data_frame))


# In[ ]:



# 1 - Filtering Pandas data frame
x = data['budget']>200000000     # There are only 3 films who have higher budget value than 200000000
data[x]


# In[ ]:


# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['budget']>200000000) & (data['popularity']>100)]


# In[ ]:




