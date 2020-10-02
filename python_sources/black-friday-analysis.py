#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpt
import matplotlib.pyplot as plt
import seaborn as sb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/BlackFriday.csv")


# ## basic analysis of the data

# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.isna().any()


# In[ ]:


print("Product_Category_1", df['Product_Category_2'].unique())
print("Product_Category_1", df['Product_Category_3'].unique())


# In[ ]:


df.fillna(0,inplace=True)


# In[ ]:


df.isna().any()


# In[ ]:


sb.countplot(df['Gender'])


# In[ ]:


sb.countplot(df['Age'], hue=df['Gender'])


# In[ ]:


df['combined_G_M'] = df.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)
print(df['combined_G_M'].unique())


# In[ ]:


sb.countplot(df['Age'],hue=df['combined_G_M'])


# In[ ]:


df['combined_G_M'] = df.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)
print(df['combined_G_M'].unique())


# In[ ]:


sb.countplot(df['Age'],hue=df['combined_G_M'])


# In[ ]:


sb.countplot(df['Product_Category_2'],hue=df['combined_G_M'])


# In[ ]:




