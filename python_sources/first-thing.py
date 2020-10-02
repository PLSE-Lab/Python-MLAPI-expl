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


data[data['title']=='Avatar']


# In[ ]:


data [data['vote_average']==data['vote_average'].max()]


# In[ ]:


data[data['popularity']==data['popularity'].min()]


# In[ ]:


#correlation plot
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True , linewidths=.1 , fmt='.1f', ax=ax)
plt.show()


# In[ ]:


data.vote_average.plot(color ='red',kind='line',label='vote',grid=True,linewidth=1,linestyle='-')
data.runtime.plot(label='runtime',grid=True,linewidth=1,linestyle=':',alpha=0.7)
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()


# In[ ]:


plt.scatter(data.vote_count,data.vote_average,color='red',alpha=0.5)
plt.xlabel('Count')
plt.ylabel('Vote Average')
plt.title('Scatter Plot')
plt.show()


# In[ ]:


data.vote_average.plot(kind='hist',bins=30,figsize=(8,8),color ='black',alpha=0.6)
plt.title('Histogram Plot')
plt.xlabel('Vote Average')
plt.ylabel ('Total numbers of given points')

plt.show()

