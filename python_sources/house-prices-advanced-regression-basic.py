#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv('../input/train.csv')
data.head()


# In[3]:


data.info()


# In[4]:


sns.distplot(data['SalePrice'])


# In[5]:


data['SalePrice'].skew()


# In[6]:


data['SalePrice'].kurt()


# In[7]:


var = 'GrLivArea'
sns.scatterplot(data[var], data['SalePrice'], marker='o')


# In[8]:


var = 'TotalBsmtSF'
sns.scatterplot(data[var], data['SalePrice'], marker='o')


# In[9]:


sns.boxplot(data['OverallQual'], data['SalePrice'])


# In[10]:


var = 'YearBuilt'
sns.scatterplot(data[var], data['SalePrice'])


# In[11]:


corr_mat = data.corr()
plt.subplots(figsize=(12,12))
sns.heatmap(corr_mat, cmap='YlGnBu')


# # Features that look promising
# - OverallQual
# - YearBuilt
# - TotalBsmtSF
# - 1stFlrSF
# - GrLivArea
# - FullBath
# - TotRmsAbvGrd
# - GarageCars
# - GarageArea
# 

# In[12]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corr_mat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': k}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[13]:


cols = corr_mat.nlargest(k, 'SalePrice')
cols


# In[14]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data[cols], height = 2.5)
plt.show();


# In[15]:


df_null = data.isnull().sum()
df_null = df_null[df_null != 0].sort_values(ascending = False)
df_null = df_null/len(data)*100
df_null
# Percentage of missing/null values


# In[16]:


cols_drop = list(df_null.index[:6].values)
for col in cols_drop:
    data = data.drop(col, axis=1)


# In[17]:


#Electrical remove one missing value
ind = data.loc[data['Electrical'].isnull()].index
data = data.drop(ind)


# In[18]:


cols_drop


# In[19]:


# Fixing skewness
from scipy.stats import norm
from scipy import stats
data.head()
# sns.distplot(, fit=norm);
# fig = plt.figure()
# res = stats.probplot(data['SalePrice'], plot=plt)


# In[20]:


#applying log transformation
data['SalePrice'] = np.log(data['SalePrice'])
sns.distplot(data['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(data['SalePrice'], plot=plt)


# In[31]:


df = pd.get_dummies(data[cols[1:]])
df.head()


# In[35]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(df, data['SalePrice'])
rf.score(df, data['SalePrice'])


# In[55]:


data_test = pd.read_csv('../input/test.csv')
# df_test.isnull().sum().sort_values(ascending=False)


# In[47]:


df_test = pd.get_dummies(df_test[cols[1:]])
df_test.head()


# In[58]:


df_test.loc[df_test['GarageCars'].isnull(), 'GarageCars'] = 1.0
df_test.loc[df_test['TotalBsmtSF'].isnull(), 'TotalBsmtSF'] = 896


# In[63]:


df_test.isnull().sum()


# In[76]:


prices = rf.predict(df_test)
prices = np.exp(prices)
result = pd.DataFrame()
result['Id'] = data_test['Id']
result['SalePrice'] = prices


# In[77]:


result.head()


# In[ ]:




