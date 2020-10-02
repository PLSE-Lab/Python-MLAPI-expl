#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/tmdb_5000_movies.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


data.head(10)


# In[ ]:


data.columns


# In[ ]:


data.tail()


# In[ ]:


data.budget.plot(kind = 'line', color = 'red',label = 'budget', linewidth=1, alpha=1, grid=True, figsize = (30,30))
data.revenue.plot(kind = 'line', color = 'green', label = 'revenue', linewidth=1, alpha=0.5, grid=True, figsize = (30,30))
plt.legend(loc = 'upper left')
plt.xlabel('budget')
plt.ylabel('revenue')
plt.title('Budget vs Revenue')
plt.show()


# In[ ]:


data.plot(kind = 'scatter', x = 'budget', y = 'vote_average', alpha = 1, color = 'green', figsize = (30,30))
plt.xlabel('budget')
plt.ylabel('vote_average')
plt.title('Comparison')
plt.show()


# In[ ]:


data.vote_average.plot(kind = 'hist', bins= 100, figsize = (30,30))
plt.xlabel('vote rating')
plt.show()


# In[ ]:


va = data['vote_average'] > 8.0
data[va]


# In[ ]:


type(data)


# In[ ]:


data1 = data.drop(['genres','homepage','id','keywords','runtime','status','tagline','vote_count','overview','production_companies','production_countries','spoken_languages'],axis=1)
data2 = data1[va]
data2


# In[ ]:


data2.sort_values(by=['budget'])


# In[ ]:


data2.sort_values(by = 'revenue')


# In[ ]:


data2.sort_values(by='vote_average')


# In[ ]:




