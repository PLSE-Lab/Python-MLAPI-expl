#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# data read / import
data = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')


# In[ ]:


# information about data
data.info()


# In[ ]:


# columns of data 
data.columns


# In[ ]:


data.corr()


# In[ ]:


# correlation map
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),annot=True, linewidths=.7,fmt ='.1f',ax=ax)
plt.show()


# In[ ]:


# 10 movies about of dataset
data.head(10)


# In[ ]:


# line plot 
# first of all check columns 'data.columns'
# budget vs revenue 
data.budget.plot(kind = 'line', color = 'r', label = ' budget', linewidth=1,alpha =0.5  ,grid = True, linestyle = ':')
data.revenue.plot(color = 'b',label = 'revenue',linewidth=1,alpha = 0.5,grid = True, linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel(' budget ')
plt.ylabel(' revenue ')
plt.title('budget vs revenue')
plt.show()


# In[ ]:


# x vote_count , y popularity
data.plot(kind='scatter',x='vote_count',y='popularity',alpha=0.5,color='red')
plt.xlabel('vote_count')
plt.ylabel('popularity')
plt.title('Vote Count / Popularity Scatter plot')
plt.show()


# In[ ]:


# alternative scatter plot
plt.scatter(data.vote_count,data.popularity,color='red')
plt.show()


# In[ ]:


#histogram
# values of revenue 
data.revenue.plot(kind = 'hist',bins = 100,figsize = (10,10))
plt.show()


# In[ ]:


# Vote average bigger than 8 
x = data['vote_average']>8
data[x]


# In[ ]:


# Vote average bigger than 8 and Vote count higher than 10.000 
data[(data['vote_average']>8) & (data['vote_count']>10000) ]


# In[ ]:


for index,value in data[['title']][0:10].iterrows():
    print(index," : ",value)

