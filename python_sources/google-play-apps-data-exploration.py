#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/googleplaystore.csv')


# **Let's Explore the data**

# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df = df.dropna() # dropping the null values
df.shape


# **Let's start exploring the data**

# In[ ]:


p = sns.countplot(x='Category', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90) # to rotate the overlapping labels in x-axis


# In[ ]:


plt.figure(figsize=(25, 10))
p = sns.countplot(x='Genres', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)


# In[ ]:


p = sns.countplot(x='Type', data=df)


# In[ ]:


p = sns.countplot(x='Content Rating', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)


# In[ ]:


p = sns.countplot(x='Installs', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)


# In[ ]:


plt.figure(figsize=(12, 5))
p = sns.countplot(x='Android Ver', data=df)
_ = plt.setp(p.get_xticklabels(), rotation=90)

