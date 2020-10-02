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


data = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


data.plot(kind='line', x='date', y='n_killed',alpha =0.8, color = 'r')
plt.xlabel('date')
plt.ylabel('number_killed')
plt.show()


# In[ ]:


data.plot(kind='line', x='date', y='n_injured',alpha =0.8, color = 'g')
plt.xlabel('date')
plt.ylabel('number_injured')
plt.show()


# In[ ]:


data.plot(kind='scatter', x='latitude', y='longitude', color='blue', alpha = 0.6)
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()


# In[ ]:


data.n_injured.plot(kind='hist', figsize=(12,12))
plt.show()


# In[ ]:


data.describe()


# In[ ]:


x = data.n_killed > 5
print(data[x])


# In[ ]:


data[(data['n_killed']>5) & (data['n_injured']>5)]


# In[ ]:


y = data.loc[:,'n_guns_involved']
print(y)

