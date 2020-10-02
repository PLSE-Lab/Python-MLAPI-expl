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


data = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data.head()


# In[ ]:


data.columns
plt.scatter(data.price,data.year, color="red", alpha = 0.5)
plt.show()


# In[ ]:


#Line
data.price.plot(kind = 'line', color = 'g',label = 'Price',linewidth=1,alpha = 0.5, grid = True )
data.year.plot(kind = 'line', color = 'r',label = 'Year',linewidth=1, alpha = 0.5, grid = True)
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# In[ ]:


# Scatter Plot 
# x = price, y = year columns
data.plot(kind='scatter', x='price', y='year',alpha = 0.15,color = 'black', edgecolors = 'none')
plt.xlabel('Price')              # label = name of label
plt.ylabel('Year')
plt.title('Price Year Scatter Plot')            # title = title of plot


# In[ ]:


# Histogram
data.price.plot(kind = 'hist',color = 'purple', bins = 150,figsize = (10,10), histtype='step', align = 'right')
plt.show()


# In[ ]:


data = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')


# In[ ]:


x = data['price']>50000
data[x]


# In[ ]:


data[(data['price']>50000) & (data['year']>2018)]

