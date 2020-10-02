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


# I am importing data.

# In[ ]:


data = pd.read_csv('../input/vgsales.csv')


# Let's display the columns.

# In[ ]:


data.columns


# I want to look at the types of datas and number of colums and rows in the dataset.

# In[ ]:


data.info()


# I would like to have a general knowledge of the values in the dataset. Here, the mean values can help me to draw some conclusions.

# In[ ]:


data.describe()


# It may be good to see the first 4 indexes of the table.

# In[ ]:


data.head()


# Which year has most been sold?

# In[ ]:


data.Year.plot(kind='hist', bins=50)
plt.show()


# I used a for loop to find out how many Nintendo games are in the table.

# In[ ]:


count = 0
for index,value in data [['Publisher']][0:].iterrows():
    if value[0] == 'Nintendo':
        count = count + 1
        print(index, "=" ,value[0])
print("Number of Nintendo in toplist", count)


# I used the correlation operator to view the relationship of datas with each other

# In[ ]:


data.corr()


# I am creating heatmap for easier understandability.

# In[ ]:


f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# It seems that most of the global sales are covered by NA sales.

# I am finding companies that have recently sold a lot to understand the most successful company and I apply a filter for this.

# In[ ]:


data[np.logical_and(data['Year']>1999, data['Global_Sales']>25 )]


# Apparently the most successful gaming company of recent years is Nintendo.
