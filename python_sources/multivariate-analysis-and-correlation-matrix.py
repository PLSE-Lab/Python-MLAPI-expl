#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm


# ### Read training set

# In[38]:


df_train = pd.read_csv('../input/train.csv')
df_train.columns


# ## Univariate analysis

# Displays the statistic details or descriptive statistics of each variable

# In[39]:


df_train.describe().T


# ### Explore outcome

# In[40]:


df_train['SalePrice'].describe()


# In[57]:


df_train['SalePrice'].plot(kind="hist")


# In[41]:


sns.distplot(df_train['SalePrice'], fit = norm)


# In[42]:


df_train['SalePrice'].skew()


# In[43]:


df_train['SalePrice'].kurt()


# In[44]:


df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
sns.distplot(df_train['SalePrice'], fit = norm)


# In[45]:


df_train['GrLivArea'] = np.log1p(df_train['GrLivArea'])
sns.distplot(df_train['GrLivArea'], fit=norm)
df_train['GrLivArea'].kurt()


# In[46]:


total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data['Total'] > 0]


# In[47]:


df_train = df_train[missing_data[missing_data['Percent'] < 0.15].index]


# ## Bivariate analysis

# In[48]:


df_train.hist(bins=50, figsize=(30,20));


# In[53]:


var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice')


# In[56]:


var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
plt.xticks(rotation=90);


# In[49]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


# ## Multivariate analysis

# correlation matrix

# In[66]:


corrmat = df_train.corr(method='spearman')
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)


# In[67]:


#correlation matrix
corrmat = df_train.corr(method='spearman')
cg = sns.clustermap(corrmat, cmap="YlGnBu", linewidths=0.1);
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
cg


# In[69]:


#saleprice correlation matrix
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, ax=ax, cmap="YlGnBu", linewidths=0.1, yticklabels=cols.values, xticklabels=cols.values)


# In[ ]:




