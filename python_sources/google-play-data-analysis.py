#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


playstore = pd.read_csv('../input/googleplaystore.csv', header="infer")
playstore.head()


# ### Let's start with checking basic info

# In[ ]:


playstore.describe()


# In[ ]:


playstore.info()


# In[ ]:


playstore.isnull().any()


# In[ ]:


playstore.dropna(inplace=True)


# ### We should probably convert Price to float type

# In[ ]:


import re

def fixPrice(column):
    return re.sub('([a-zA-Z]+|\$)', '', column)

# Convert price to float. 
playstore['Price'] = playstore['Price'].apply(fixPrice)
playstore['Price'] = pd.to_numeric(playstore['Price'])

# Get most expensive app.
max_val = playstore['Price'].max()
playstore[playstore['Price'] == max_val]


# ### What a surprise the most expensive app costs $400 !

# In[ ]:


len(playstore['Category'].unique())


# In[ ]:


from pylab import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

top_five_categories = playstore['Category'].value_counts().head().rename_axis('categories').reset_index(name='counts')
g = sns.barplot(top_five_categories.categories, top_five_categories.counts)
g = sns.color_palette("husl", 3) 
plt.title("Top 5 categories by number of apps") 

# Adjusting figure's size
rcParams['figure.figsize'] = 5, 10
plt.show(g)


# ### Questions to answer
# 
# * What is the mean rating per app category ?
# * What is the mean rating per app **category** per **Type (free or paid)** ?
# * What is the average price per app category ?
# * Best rated apps are paid, true story ?

# ### Let's see number of paid vs free apps

# In[ ]:


playstore['Type'].value_counts()


# In[ ]:


category_rating = playstore.groupby('Category')['Rating'].mean().sort_values(ascending=False)
category_rating


# In[ ]:


import seaborn as sns

sns.set()
plt.rcParams['figure.figsize'] = [25, 10]
sns.barplot(x=category_rating[:10].index, y=category_rating[:10].get_values())


# In[ ]:


paid_mean_price =  playstore[playstore['Type'] == 'Paid'].groupby('Category')['Price'].mean().sort_values(ascending=False)
paid_mean_price


# In[ ]:


sns.set()
sns.barplot(x=paid_mean_price[:10].index, y=paid_mean_price[:10].get_values())


# ### Financial apps seem to be the most expensive here, with a mean price of $187 !

# In[ ]:


pivoted = playstore.pivot_table(index='Category', columns='Type', values='Rating', aggfunc='mean')
pivoted


# **Those NaN are present because there are some categories which don't have have Paid apps.**
# 
# ** We can now answer the final question which is: are paid app more rated than free apps ? **

# In[ ]:


pivoted.mean()


# ### We can see here that both Types are almost equally rated

# In[ ]:




