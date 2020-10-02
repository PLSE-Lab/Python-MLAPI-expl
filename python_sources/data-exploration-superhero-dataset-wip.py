#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/superhero-set"))

# Any results you write to the current directory are saved as output.


# In[4]:


df_information = pd.read_csv("../input/superhero-set/heroes_information.csv")
df_information.head()
df = df_information.copy()


# # Know your data
# Extract some information to help you deal with missing data

# In[ ]:


print(df.shape)
df.info()


# From info method, you can check columns names, type of data for each column, and non-null values count, from which it can be count all null values. But for a more clean information on that, we can use the isnull method:

# In[ ]:


df.isnull().sum()


# To calculate the percentage of missing data:

# In[5]:


df.isnull().sum().sum()/df.shape[1]


# What about other values that can be considered as empty values, like string columns with empty strings or values with "-"?

# In[11]:


# this way even empty strings are counted as null 
df.replace('', np.nan).isnull().sum()
df.replace('[-|]', np.nan, regex=True).isnull().sum()


# Or negative values for height and weight, what wouldn't make sense

# In[ ]:


print(df[df['Height'] < 0].shape)
df[df['Weight'] < 0].shape


# Those values are better being treated as NaN

# In[ ]:


df.loc[df['Height'] < 0] = np.nan
df.loc[df['Weight'] < 0] = np.nan


# Is important to look for duplicate rows! The method duplicated can do the trick.

# In[ ]:


df.duplicated().sum()


# But sometimes is more important to look for duplicated values in specific columns, like Heros name, in that case. You can check how many duplicated values, see a list of them, or see the entire rows they appear, to have a better look at wha is going on.

# In[ ]:


print(df['name'].duplicated().sum())
print(df[df['name'].duplicated(keep=False)]['name'].unique())
print(df[df[['name']].duplicated(keep=False)])


# In[ ]:


print(df[['name', 'Gender', 'Publisher']].duplicated().sum())
df[df[['name', 'Gender', 'Publisher']].duplicated(keep=False)]


# For numeric values, descirbe method is quite handy!

# In[ ]:


df.describe()


# Data frame can be separeted in two subdatasets, with numerical an non-numerical values:

# In[ ]:


df_num = df.select_dtypes(include=['float64', 'int64'])
df_obj = df.select_dtypes(include=['object'])


# For categorical data, we can list categories and count the number of entries in each categorie:

# In[ ]:


print(df['Gender'].unique())
df['Gender'].value_counts()


# Plotting data can be helpfull, and we are doing that with Seaborn

# In[ ]:


import seaborn as sns
sns.countplot(df['Gender'])


# Its a good thing to check data distribution by a histogram, for numeric values

# In[ ]:


sns.distplot(df['Height'], bins=10, kde=False)


# In[ ]:


sns.distplot(df['Weight'], bins=10, kde=False)


# Correlation (WIP)

# ### References: 
# https://towardsdatascience.com/be-a-more-efficient-data-scientist-today-master-pandas-with-this-guide-ea362d27386
# https://towardsdatascience.com/10-python-pandas-tricks-that-make-your-work-more-efficient-2e8e483808ba
