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


data = pd.read_csv ('../input/tmdb_5000_movies.csv')


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.columns


# In[ ]:


data.corr()


# In[ ]:


data.head(10)


# In[ ]:


data[data['title']=='Harry Potter and the Half-Blood Prince']


# In[ ]:


data [data['budget']==data['budget'].min()]


# In[ ]:


data[data['revenue']==data['revenue'].max()]


# In[ ]:


#correlation plot
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True , linewidths=.1 , fmt='.1f', ax=ax)
plt.show()


# In[ ]:


data.vote_count.plot(color ='red',kind='line',label='vote_count',grid=True,linewidth=1,linestyle='-')
data.revenue.plot(label='revenue',grid=True,linewidth=1,linestyle=':',alpha=0.7)
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


# In[ ]:


plt.scatter(data.budget,data.revenue,color='black',alpha=0.1)
plt.xlabel('budget')
plt.ylabel('revenue')
plt.title('Scatter Plot')
plt.show()


# In[ ]:


data.budget.plot(kind='hist',bins=10,figsize=(12,8),color ='blue',alpha=0.9)
plt.title('Histogram Plot')
plt.xlabel('budget')
plt.ylabel ('Frequence')

plt.show()

