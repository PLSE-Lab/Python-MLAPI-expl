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


data = pd.read_csv('../input/2017.csv')
data.info()


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize = (20,20))
 #(name of table, values at mid, linewidth, decimal value, shape)
sns.heatmap(data.corr(), annot = True, linewidths =.5, fmt='.1f', ax=ax)
plt.show()


# In[ ]:


data.columns


# Dataframe column names are not useful for the plotting processes. So I have to change dots(' . ') for use columns names.

# In[ ]:


data_cols = data.columns.str.replace('.','_')
data_cols


# In[ ]:


data.columns = data_cols
data.columns
data.head(20)


# In[ ]:


#line plot
data.Freedom.plot(kind ='line', color='red', label='Freedom', linewidth = 1, alpha=0.8, grid = True, linestyle = ':', figsize=(8,8))
data.Trust__Government_Corruption_.plot(kind = 'line', color='blue', label = 'Trust Government Corruption', linewidth = 1, alpha = 0.8, grid = True, linestyle = '-.')
plt.legend()
plt.xlabel='x axis'
plt.ylabel='y axis'
plt.title = 'Freedom & Trust Government Corruption Line Plot'
plt.show()


# In[ ]:


#scatter plot
data.plot(kind = 'scatter', x = 'Happiness_Score', y = 'Freedom' , alpha = 0.6, color = 'black', grid = True, figsize=(8,8))
plt.xlabel = 'Happiness Score'
plt.ylabel = 'Freedom'
plt.title='Happiness Score & Freedom Scatter Plot'
plt.show()


# In[ ]:


#histogram plot
data.Happiness_Score.plot(kind='hist', bins=40, color = 'blue', alpha = 0.5, figsize=(8,8))
plt.xlabel='Happiness Score'
plt.show()


# I am executing different filters to table and lets see the results.

# In[ ]:


filter1 = data.Happiness_Score > data.Happiness_Score.mean()
data[filter1]


# In[ ]:


filter2 = data.Trust__Government_Corruption_ < 0.3
data[filter2]


# I executed two different filters at the same time. I used Numpy library at processing.

# In[ ]:


data[np.logical_and(filter1,filter2)]

