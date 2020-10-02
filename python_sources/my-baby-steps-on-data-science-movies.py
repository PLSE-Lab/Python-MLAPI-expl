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
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


credits = pd.read_csv("../input/tmdb_5000_credits.csv")
movies = pd.read_csv("../input/tmdb_5000_movies.csv")


# In[ ]:


credits.info()


# In[ ]:


credits.head(10)


# In[ ]:


movies.info()


# In[ ]:


movies.head(10)


# In[ ]:


movies.columns


# In[ ]:


good_films = np.logical_and(movies['vote_average']>7.0,movies['vote_count']>5000)
movies[good_films]


# In[ ]:


movies.vote_average.plot(kind = 'line', color = 'g',label = 'Vote Average',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
movies.vote_count.plot(color = 'b',label = 'Vote Count',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


movies.plot(kind='scatter', x='vote_count', y='vote_average',alpha = 0.5,color = 'blue')
plt.xlabel('Vote Count')              # label = name of label
plt.ylabel('Vote Average')
plt.title('Vote Average & Count Scatter Plot')            # title = title of plot


# In[ ]:


movies.vote_average.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


goods = movies[good_films]
for index,value in goods[['original_title']][0:2].iterrows():
    print(value)


# In[ ]:


dict = {}

a=0
for i in goods['original_title']:
    dict[a]=i
    a+=1
    if a>=goods.size:
        break

dict


# In[ ]:




