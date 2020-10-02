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


data = pd.read_csv('../input/AppleStore.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt='1f',ax=ax)
plt.show()


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


# Line Plot
data.user_rating_ver.plot(kind='line', color='red',label='current',linewidth=1,alpha=.5,grid=True,linestyle=':',figsize=(18,10))
data.user_rating.plot(kind='line',color='green',label='all',linewidth=1,alpha=.5,grid=True,linestyle='-.',figsize=(18,10))
plt.legend('upper right')
plt.show()


# In[ ]:


# Scatter Plot
# x=price , y=user_rating_ver
data.plot(kind='scatter',x='price',y='user_rating_ver',alpha=.5,color='red',figsize=(18,4))
plt.xlabel=('user rating')
plt.ylabel=('user_rating_ver')
plt.title=('User Rating by Price')
plt.show()


# In[ ]:


#Histogram
data.user_rating.plot(kind='hist',bins=20,grid=True)
plt.show()


# In[ ]:


filter_1 = data.user_rating>2.5
filter_2 = data.user_rating_ver<4
filter_3 = data.price>9.99
data_1=data[filter_1 & filter_2 & filter_3]
data_1.loc[:,['id','currency','price','user_rating','user_rating_ver']]


# In[ ]:


data_1.shape


# First of all, thank you for reading my notebook. It is my first kernel. Please comment me for improve myself. 
