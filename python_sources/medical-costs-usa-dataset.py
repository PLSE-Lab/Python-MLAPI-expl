#!/usr/bin/env python
# coding: utf-8

# In[6]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


df = pd.read_csv('../input/insurance.csv')
df.head()


# # Exploratory Data Analysis

# In[60]:


g = sns.catplot(data=df, kind='count', x='sex', palette='PRGn')


# In[40]:


g=sns.catplot(data=df, x='smoker', kind='count', palette='PRGn')


# In[59]:


g=sns.catplot(data=df, x='age', kind='count', palette='PRGn').set_xticklabels(rotation=90)


# In[56]:


g=sns.catplot(data=df, x='children', kind='count', palette='PRGn')


# In[76]:


g=sns.catplot(data=df, kind='count', x='region')


# The data set:
# - is skewed towards non-smokers
# - is skewed young: the greatest number is 18-19 yrs old
# - majority have no children 

# ## Do smokers pay more in medical costs?

# In[66]:


g=sns.catplot(data=df, kind='swarm', x='age', y='charges', hue='sex', col='smoker', palette='Spectral').set_xticklabels(rotation=90)


# - Overall smokers experience higher medical costs

# ## Does number of children influence medical costs?

# In[55]:


g=sns.catplot(data=df, kind='swarm', x='age', y='charges', hue='children').set_xticklabels(rotation=90)


# In[52]:


df.columns


# In[33]:


g=sns.catplot(data=df, kind='box', x='children', y='charges').set_xticklabels(rotation=90)


# 

# In[67]:


df.columns


# ## Does region influence medical costs?

# In[75]:


g=sns.catplot(data=df, kind='box', x='region', y='charges')


# - Region does not seem to influence medical costs 
