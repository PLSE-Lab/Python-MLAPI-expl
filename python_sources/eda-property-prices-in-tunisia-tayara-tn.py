#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/property-prices-in-tunisia/Property Prices in Tunisia.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


plt.figure(figsize=(15,7))
sns.countplot(df['category'])


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['room_count'])
plt.xlabel("Room Count")


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['bathroom_count'])
plt.xlabel("Bathroom Count")


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['type'])
plt.xlabel("Type Count")


# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(df['city']).set_xticklabels(sns.countplot(df['city']).get_xticklabels(),rotation="90")


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(y=df['city'],order=df.city.value_counts().index)


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(y=df['region'],order=df.region.value_counts().index)


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(y=df['category'],order=df.category.value_counts().index)


# In[ ]:


sns.factorplot(data=df,x='log_price', y='city', hue='type', col='category',kind='bar', col_wrap=3)


# Working on it

# In[ ]:




