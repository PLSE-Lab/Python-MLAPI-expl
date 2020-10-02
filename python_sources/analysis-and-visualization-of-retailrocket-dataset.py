#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# **Category Analysis**

# In[4]:


categories  = pd.read_csv('../input/category_tree.csv')
print(categories.categoryid.unique().size)
categories.describe()


# In[27]:


rootcat = categories[categories.parentid.isnull()]
print('number of root categories : '+ str(categories[categories.parentid.isnull()].size))
rootcat.head()


# In[31]:


subcat = categories[categories.parentid.isin(rootcat.categoryid) & categories.parentid.notnull()]
print('number of subcategories : '+ str(subcat.size))
subcat.head()


# In[32]:


subsubcat = categories[categories.parentid.isin(subcat.categoryid)]
print('number of subsubcategories : '+ str(subsubcat.size))
subsubcat.head()


# In[33]:


ssubsubcat = categories[categories.parentid.isin(subsubcat.categoryid)]
print('number of ssubsubcategories : '+ str(ssubsubcat.size))
ssubsubcat.head()


# In[35]:


events  = pd.read_csv('../input/events.csv')
print(events.shape)
print(events['event'].unique())
events.describe()


# In[56]:


data = events.event.value_counts()
indexes = data.index
values = data.values
explode = (0.15, 0, 0.1)  
colors = ['lightblue','darkseagreen','lightcoral']
plt.subplots(figsize=(8,8))
# Plot
plt.pie(values, labels=indexes,startangle=0, autopct='%.1f%%', explode=explode,colors=colors)
 
plt.title("Data proportion of user's events in Retailrocket Dataset")
plt.show()


# Combining the properties into 1 dataframe

# In[58]:


props1 = pd.read_csv('../input/item_properties_part1.csv')
props2 = pd.read_csv('../input/item_properties_part2.csv')
props = pd.concat([props1, props2])
print(props.shape)


# In[59]:


props.head()


# In[61]:


props.itemid.unique().size


# In[ ]:




