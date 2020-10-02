#!/usr/bin/env python
# coding: utf-8

# # How to deal with Missing Values in Machine Learning Problems ?

# Most of the Machine Learning Problems contains lots of missing values that create a hurdle while training a Machine Learning 
# Model. In this notebook you will find some common ways to deal with these missing values. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Imporing the required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Import Train and Test files

train=pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train.head(10)


# In[ ]:


train.info()


# In[ ]:


#Ploting the missing values using Heatmap

plt.figure(figsize = (10,6))
sns.heatmap(train.isnull())


# In[ ]:


# To find the number of Missing Values in each Columns

train.isnull().sum()


# Its better to drop columns which have more than 50% missing value than filling it. Because it may lead our model in wrong direction and may have huge errors in prediction.

# In[ ]:


#Dropping columns which have more than 50% Missing Values

train=train.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)  
test = test.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)


# In[ ]:


train.isnull().sum()


# In[ ]:


# Separating Numerical and Categorical columns

train_num_cols = train.select_dtypes(exclude='object').columns
train_cat_cols = train.select_dtypes(include='object').columns


test_num_cols = test.select_dtypes(exclude='object').columns
test_cat_cols = test.select_dtypes(include='object').columns


# In[ ]:


train_num_cols


# In[ ]:


train_cat_cols


# In[ ]:


#Filling Missing Values (Numerical Features) using mean of all non null values.

for i in range(0, len(train_num_cols)):
    train[train_num_cols[i]] = train[train_num_cols[i]].fillna(train[train_num_cols[i]].mean())
    
for i in range(0, len(test_num_cols)):
    test[test_num_cols[i]] = test[test_num_cols[i]].fillna(test[test_num_cols[i]].mean())


# In[ ]:


#Filling Missing Values (Categorical Features) using mode of all non null values.

for i in range(0, len(train_cat_cols)):
    train[train_cat_cols[i]] = train[train_cat_cols[i]].fillna(train[train_cat_cols[i]].mode()[0])
    
for i in range(0, len(test_cat_cols)):
    test[test_cat_cols[i]] = test[test_cat_cols[i]].fillna(test[test_cat_cols[i]].mode()[0])


# Note: To handle missing values of Categorical Features we do not use mean/median because we cannot find mean/median of strings.
#       while any of mean/median/mode can be used for Categorical Feature.

# # **Give it a try**

# ### Upvote if you like this notebook and feel free to ask your doubts in commnet section.
# ### Thank You:)
