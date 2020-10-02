#!/usr/bin/env python
# coding: utf-8

# In[90]:


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


# In[91]:


df = pd.read_csv('../input/scotch_review.csv')
df.head()


# In[92]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[93]:


df.head()


# In[94]:


df.dtypes


# In[95]:


df.shape


# In[96]:


df.category.value_counts()


# In[97]:


print(df.price.max())


# In[98]:


print(df.price.min())


# In[99]:


set(df.currency)


# In[100]:


df['review.point'].value_counts()[:10]


# In[101]:


df[['category','review.point','price']].sort_values('review.point',ascending=False)[:10]


# In[102]:


df[('category')].value_counts().plot(kind='bar')


# more coming soon, if you like it please upvote for me please.`
