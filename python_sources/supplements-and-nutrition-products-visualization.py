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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


df = pd.read_csv('../input/bodybuilding_nutrition_products.csv')


# In[ ]:


df.info()


# In[ ]:


df.isnull().any()


# In[ ]:


df.fillna(0,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.drop(['link','product_description','verified_buyer_number'], axis=1,inplace = True)


# In[ ]:


df.head()


# In[ ]:


print(df['brand_name'].unique())
print(df['product_category'].unique())
print(df['product_name'].unique())
print(df['top_flavor_rated'].unique())


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


corr = df.corr()
plt.figure(figsize=(12,6))
sns.heatmap(corr,annot=True)
plt.show()


# **Does price affect overall product rating? Is price negatively correlated with overall product rating?**
# 
# There is almost zero correlation b/w price and Overall product rating, means price won't affect overall product rating.

# In[5]:


plt.figure(figsize=(12,6))
sns.pairplot(df)
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot('product_category',data=df,palette='viridis', order = df['product_category'].value_counts().index)
plt.xticks(rotation=90,ha='right')
plt.tight_layout()
plt.show()


# In[ ]:


df.columns


# In[ ]:


df['product_category'].replace([0],['unnamed'], inplace=True);


# In[ ]:



brand_df = df[['average_flavor_rating', 'brand_name', 'number_of_flavors','product_category']]
brand_df.head()

