#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np # linear algebra
import pandas as pd # data processing
import os
import matplotlib.pyplot as plt # charts and graphs
import seaborn as sns # styling & pretty colors

dataset = pd.read_csv('../input/PGA Tour 2010-2018.csv')
dataset.info()
dataset.head()
dataset.corr()


# 
# 
# 
# **Removing unwanted statistics and limiting the data by year**

# In[20]:


dataset = dataset[dataset.Variable.str.contains('%')]

df_test = dataset[dataset.Season == 2018]
df_train = dataset[dataset.Season != 2018]


# In[21]:


df_train.head()


# **Pivoting based on *Variable* for both *Player Name* and *Season* columns, resulting in a MultiIndex dataframe. **

# In[22]:


# Transposed based on key value pairs
df = df_train.set_index(['Player Name', 'Variable', 'Season'])['Value'].unstack('Variable').reset_index()
df_train = df_train.set_index(['Player Name', 'Variable', 'Season'])['Value'].unstack('Variable').reset_index()


print("original column count:\t" + str(len(dataset.columns)))
print("     new column count:\t" + str(len(df_train.columns)))


# In[23]:


df_train.columns


# In[24]:


#df_train = df_train.dropna(thresh=df_train.shape[1]*0.5,how='all',axis=1)


# In[25]:


#df_train.columns


# In[26]:


#print (df_train.isnull().mean() * 100)
df_train = df_train.loc[:, df_train.isnull().mean() <= .9]
print("    new column count:\t" + str(len(df_train.columns)))
print("current row count:\t" + str(len(df_train)))


# In[34]:


df_train.dropna()
df.dtypes


# In[32]:


df_train = df_train.apply(pd.to_numeric, errors='coerce')
df_train.dtypes


# <a href="https://www.kaggle.com/tylerguy/crypto-analysis"><h1>PART 2
# https://www.kaggle.com/tylerguy/crypto-analysis </h1></a>

# ![https://scontent-dfw5-2.cdninstagram.com/vp/9a03952f81e6d5e3db74b73edd2707b9/5D3670F8/t51.2885-15/e35/40031123_317382939023601_6297048893527949312_n.jpg?_nc_ht=scontent-dfw5-2.cdninstagram.com](https://scontent-dfw5-2.cdninstagram.com/vp/9a03952f81e6d5e3db74b73edd2707b9/5D3670F8/t51.2885-15/e35/40031123_317382939023601_6297048893527949312_n.jpg?_nc_ht=scontent-dfw5-2.cdninstagram.com)

# In[ ]:





# In[ ]:




