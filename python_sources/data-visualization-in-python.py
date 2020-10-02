#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


Data = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
Data.head()


# In[ ]:


Data.describe()


# In[ ]:


Data.dtypes


# In[ ]:


Data.columns


# In[ ]:


Data.tail()


# In[ ]:


Data.count()


# In[ ]:


Data.isnull().sum()


# In[ ]:


Data.isna().sum()


# In[ ]:


Data.shape


# In[ ]:


Data['shop_id'].mean()


# In[ ]:


Data['item_id'].mean()


# In[ ]:


Data['shop_id'].median()


# ****HUE IN SEABORN
# #In seaborn, the hue parameter determines which column in the data frame should be
# #used for colour encoding. ... Adding `hue="smoker" tells seaborn you want 
# #to colour the data points for smoker and non-smoker differently.
# 

# In[ ]:


Data['item_id'].median()


# In[ ]:


sns.lmplot(x='shop_id',y='item_id',data=Data,legend=True,palette='red')


# In[ ]:


sns.distplot(Data['shop_id'])


# In[ ]:


sns.distplot(Data['shop_id'],kde=False)


# In[ ]:


sns.distplot(Data['shop_id'],bins=3)


# In[ ]:


sns.distplot(Data['shop_id'],bins=2)


# In[ ]:


sns.distplot(Data['shop_id'],bins=5)


# In[ ]:


sns.distplot(Data['shop_id'],bins=10)


# In[ ]:


sns.distplot(Data['item_id'])


# In[ ]:


sns.distplot(Data['item_id'],kde=False)


# In[ ]:


sns.distplot(Data['item_id'],bins=5)


# In[ ]:


sns.distplot(Data['item_id'],bins=3)


# In[ ]:


sns.distplot(Data['item_id'],bins=10)


# #  **BAR PLOT****

# In[ ]:


sns.countplot(x='shop_id',data=Data)


# In[ ]:


sns.countplot(x='item_id',data=Data)


# GROUPED BAR PLOT

# In[ ]:


pd.crosstab(index=Data['shop_id'],columns=Data['item_id'],dropna=True)


# In[ ]:


pd.crosstab(index=Data['item_id'],columns=Data['shop_id'],dropna=True)


# # > **#BOX AND WHISKERS PLOT FOR NUMARICAL VARIABLES****
# 

# In[ ]:


sns.boxplot(x=Data['shop_id'])


# In[ ]:


sns.boxplot(y=Data['shop_id'])


# In[ ]:


sns.boxplot(x=Data['item_id'])


# In[ ]:


sns.boxplot(y=Data['item_id'])


# In[ ]:


sns.boxplot(x=Data['shop_id'],y=Data['item_id'])


# # SCATTER PLOT

# In[ ]:


plt.scatter(Data['shop_id'],Data['item_id'],c='green')
plt.title('Scatter plot of shop_id vs item_id')
plt.xlabel('shop_id')
plt.ylabel('item_id')


# # HISTOGRAM

# In[ ]:


plt.hist(Data['shop_id'])


# In[ ]:


plt.hist(Data['item_id'])


# FREQUENCY ONE WAY TABLE

# In[ ]:


plt.hist(Data['shop_id'],color='red',edgecolor='red',bins = 200)


# In[ ]:




