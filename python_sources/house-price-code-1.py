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


# Following code reads the dataset in train and test dataframes

# In[ ]:


train=pd.read_csv('../input/train.csv')


# In[ ]:


test=pd.read_csv('../input/test.csv')


# * Summary of train dataframe.
# * Model training will be done by this dataframe

# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


train.describe()


# Summarized

# Lets find out numeric and non numeric columns, and store them in separate lists

# In[ ]:


cols=train.columns
non_numeric_cols = [i for i in cols if train[i].dtype=='object']


# In[ ]:


print(non_numeric_cols)


# In[ ]:


numeric_cols=[i for i in cols if i not in non_numeric_cols]


# In[ ]:


print(numeric_cols)


# Lets now do some visualizations for numeric columns and then later we will see for categorical data, about what can be done

# In[ ]:


#importing libraris for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_clean=train[numeric_cols].dropna()

for num_col in train_clean.columns:
    plt.subplots(figsize=(12,10))
    sns.regplot(x=num_col,y='SalePrice',data=train_clean)


# In[ ]:


train.isnull().sum()


# In[ ]:


len(train)


# In[ ]:


train[['PoolQC','SalePrice']].dropna()


# In[ ]:


train[non_numeric_cols].head()


# In[ ]:


corr=train.corr()


# In[ ]:


len(numeric_cols)


# In[ ]:


cat_cols=[col for col in non_numeric_cols if train[col].nunique()<=6]


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# Missing value treatment:
# 1. first we will drop the columns having more than 60% missing data
# 2. we will drop the rows containing missing values

# In[ ]:


length=len(train)
missing_perc_cols=[col for col in train.columns if train[col].isnull().sum(axis=0)/length>0.4]
missing_perc_cols


# In[68]:


import seaborn as sns
for col in missing_perc_cols:
    plt.subplots(figsize=(12,10))
    sns.countplot(x=col,data=train)


# In[ ]:




