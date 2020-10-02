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
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/fifa_ranking.csv')


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.shape 


# In[ ]:


data.head(10)


# In[ ]:


data.tail(10)


# In[ ]:


data.describe()


# In[ ]:


data1=data["previous_points"]>250
data[data1]


# In[ ]:


data[np.logical_and(data["previous_points"]>500,data["rank_change"]>10)]


# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.5f',ax=ax)
plt.show()


# In[ ]:


data.plot(figsize=(20,20)) 


# In[ ]:


data.plot(kind = 'line', x='rank', y='previous_points', color = 'blue', linewidth=1,alpha = 0.5, grid = True, linestyle = '-.', figsize=(20,15))
plt.legend(loc='upper right')     
plt.ylabel('Previous Points')              
plt.xlabel('Rank')
plt.title('FIFA Soccer Rankings - Line Plot')
plt.show()


# In[ ]:


data.plot(kind = 'line', x='total_points', y='cur_year_avg', color = 'black', linewidth=1,alpha = 0.5, grid = True, linestyle = '--', figsize=(20,15))
plt.legend(loc='upper right')     
plt.xlabel('Total Points')              
plt.ylabel('Cur Year Average')
plt.title('FIFA Soccer Rankings - Line Plot')
plt.show()


# In[ ]:


plt.subplots(figsize=(20,10))
plt.plot(data['total_points'], data['rank'], color='r', label='Rank')
plt.plot(data['total_points'], data['last_year_avg'], color='b', label='Last Year Average')
plt.legend()
plt.title('FIFA Soccer Rankings - Scatter Plot')
plt.xlabel('Total Points')
plt.show()

