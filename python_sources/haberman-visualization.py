#!/usr/bin/env python
# coding: utf-8

# In[27]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv('../input/haberman.csv')


# In[6]:


df.head()


# In[7]:


df.columns


# In[11]:


df['age']=df['30']
df['year_op']=df['64']
df['axil_node']=df['1']
df['survival']=df['1.1']


# In[21]:


df.drop('1.1',axis=1,inplace=True)


# In[22]:


df.head()


# # How many survived?
# 
# almost 230 patients

# In[37]:


sn.countplot(x='survival',data=df)


# # how age affected survival?
# people with lower ages survived

# In[24]:


sn.boxplot(x='survival',y='age',data=df)


# # distribution of the disease with age

# In[40]:


plt.figure(figsize=(15,6))
sn.countplot(x='age',data = df)


# In[35]:


sn.distplot(a=df['age'],kde=True)


# In[36]:


df.columns


# # relation between number of auxillary nodes and the survival rate

# In[42]:


plt.figure(figsize=(10,6))
sn.violinplot(x='survival',y='axil_node',data=df)


# In[48]:


sn.stripplot(y='axil_node',x='survival',data=df,jitter=True)


# In[ ]:




