#!/usr/bin/env python
# coding: utf-8

# **Starting from the first 5000 films of the IMDB, we will examine the effect of the money spent on the production of the films on the points earned and the money earned.**

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


data = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv") #dataset import to data veriables use pandas


# In[ ]:


data.info() 


# In[ ]:


data.corr #data correlation map, but too big veriables readible impossible.


# In[ ]:


#correlation heat map
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.2f', ax=ax)
plt.show()


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.budget.plot(kind = 'line', color = 'g',label = 'Budget',linewidth=2,alpha = 0.5,grid = True,linestyle = ':',figsize = (30,12))
data.revenue.plot(color = 'r',label = 'Revenue',linewidth=2, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
# x = attack, y = defense
data.plot(kind='scatter', x='budget', y='revenue',alpha = 0.5,color = 'red',figsize = (12,12))
plt.xlabel('Budget')              # label = name of label
plt.ylabel('Revenue')
plt.title('Attack Defense Scatter Plot')            # title = title of plot


# In[ ]:


# Histogram
# bins = number of bar in figure
data.revenue.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# In[ ]:


data[(data['budget']<100000000) & (data['revenue']>1000000000)] #budget less 100M$ and revenue more 1B$


# **In general, the more money spent on films, the more money will be earned. However, it is not just the money spent that hampers good scores and high money. Likewise, the runtime is a positive factor. Making high money doesn't mean you get good points.**
