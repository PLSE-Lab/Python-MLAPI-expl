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


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/consumer_complaints.csv",low_memory=False)
df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# In[ ]:


p_product_discussions = round(df["product"].value_counts() / len(df["product"]) * 100,2)
p_product_discussions


# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(df['product'])
plt.show()


# In[ ]:


temp = df.company.value_counts()[:10]
temp


# In[ ]:


plt.figure(figsize=(20,5))
sns.barplot(temp.index,temp.values)
plt.xticks(rotation=60)
plt.show()


# In[ ]:


temp = df.state.value_counts()[:10]
temp


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(temp.index,temp.values)


# In[ ]:


temp = df.company_response_to_consumer.value_counts()
temp


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(y = temp.index, x= temp.values)


# In[ ]:


df.timely_response.value_counts()


# In[ ]:


sns.countplot(df.timely_response)


# In[ ]:


df['consumer_disputed?'].value_counts()


# In[ ]:


sns.countplot(df['consumer_disputed?'])


# In[ ]:


top5_disputed = df['company'].loc[df['consumer_disputed?'] == 'Yes'].value_counts()[:5]
top5_disputed


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x = top5_disputed,y = top5_disputed.index)
plt.show()


# In[ ]:


top5_nodispute = df['company'].loc[df['consumer_disputed?'] == 'No'].value_counts()[:5]
top5_nodispute


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(x = top5_nodispute.values,y = top5_nodispute.index)
plt.show()


# In[ ]:


df['date_received'] = pd.to_datetime(df['date_received'])
df['year_received'], df['month_received'] = df['date_received'].dt.year, df['date_received'].dt.month
df.head()


# In[ ]:


df.year_received.value_counts()


# In[ ]:


sns.countplot(df.year_received)


# In[ ]:


df.month_received.value_counts()


# In[ ]:


sns.countplot(df.month_received)


# more in pipeline.
# 
# **if you like it please upvote for me**
# 
# Thank you : )
