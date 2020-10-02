#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew


# In[6]:


HousePricing = pd.read_csv('../input/train.csv')
HousePricing.columns


# In[7]:


HousePricing.tail()


# In[8]:


HousePricing.describe()


# In[9]:


HousePricing.isnull().sum().sort_values(ascending=False)


# In[10]:


((HousePricing.isnull().sum() / HousePricing.isnull().count()) * 100).sort_values(ascending=False)


# In[11]:


HousePricing.tail()


# In[12]:


HousePricing = HousePricing.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
HousePricing.tail()


# In[13]:


fig, ax = plt.subplots()
sns.regplot('GrLivArea', 'SalePrice', data=HousePricing)
plt.xlabel('GrLivArea', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)
plt.show()


# In[14]:


HousePricing = HousePricing.drop(HousePricing[(HousePricing['GrLivArea'] > 4000) & (HousePricing['SalePrice'] < 300000)].index)
HousePricing = HousePricing.drop(HousePricing[HousePricing['SalePrice'] > 700000].index)


# In[15]:


sns.regplot('GrLivArea', 'SalePrice', data=HousePricing)
plt.show()


# In[16]:


sns.distplot(HousePricing['SalePrice'])
plt.show()


# In[17]:


HousePricing['SalePrice'] = np.log1p(HousePricing['SalePrice'])
sns.distplot(HousePricing['SalePrice'], fit=norm)


# In[18]:


stats.probplot(HousePricing['SalePrice'], plot=plt)
plt.show()


# In[19]:


corrmat = HousePricing.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()


# In[20]:


k = 10
plt.figure(figsize=(16, 16))
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(HousePricing[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[21]:


sns.regplot('OverallQual', 'SalePrice', data=HousePricing)
plt.show()


# In[22]:


plt.figure(figsize=(16, 16))
sns.regplot('YearBuilt', 'SalePrice', data=HousePricing)
plt.show()


# In[23]:


plt.figure(figsize=(13, 13))
sns.regplot('YearBuilt', 'GrLivArea', data=HousePricing)
plt.show()


# In[24]:


def roundten(year):
    return int(year / 10) * 10

HousePricing['Decade'] = HousePricing['YearBuilt'].apply(roundten)
HousePricing['Decade'].tail(10)


# In[25]:


plt.figure(figsize=(16, 16))
sns.boxplot('Decade', 'SalePrice', data=HousePricing)
plt.show()


# In[26]:


def roundhoundred(val):
    return int(val / 100) * 100

HousePricing['LivAreaTen'] = HousePricing.GrLivArea.apply(roundhoundred)
plt.figure(figsize=(15, 15))
sns.boxplot('LivAreaTen', 'SalePrice', data=HousePricing)
plt.xticks(rotation=90)
plt.show()


# In[27]:


plt.figure(figsize=(15, 15))
sns.boxplot('OverallQual', 'SalePrice', data=HousePricing)
plt.xticks(rotation=90)
plt.show()


# In[28]:


sns.boxplot('GarageCars', 'SalePrice', data=HousePricing)
plt.show()


# In[29]:


cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt']
sns.pairplot(vars=cols, data=HousePricing)
plt.show()


# In[30]:


HousePricing.isnull().sum().sort_values(ascending=False)


# In[31]:


# Clean
HousePricing = HousePricing[~(HousePricing['Electrical'].isnull())]
HousePricing = HousePricing[~(HousePricing['MasVnrArea'].isnull())]
HousePricing = HousePricing[~(HousePricing['MasVnrType'].isnull())]
HousePricing = HousePricing.drop(['LotFrontage', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageType'], 
                                axis=1)
HousePricing.isnull().sum().sort_values(ascending=False)


# In[32]:


StrangeVars = HousePricing[['BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 
              'BsmtQual']]
StrangeVars.tail(10)


# In[33]:


HousePricing['BsmtFinType2'] = HousePricing['BsmtFinType2'].fillna(HousePricing['BsmtFinType2'].mode()[0])
HousePricing['BsmtExposure'] = HousePricing['BsmtExposure'].fillna(HousePricing['BsmtExposure'].mode()[0])
HousePricing['BsmtFinType1'] = HousePricing['BsmtFinType1'].fillna(HousePricing['BsmtFinType1'].mode()[0])
HousePricing['BsmtCond'] = HousePricing['BsmtCond'].fillna(HousePricing['BsmtCond'].mode()[0])
HousePricing['BsmtQual'] = HousePricing['BsmtQual'].fillna(HousePricing['BsmtQual'].mode()[0])
HousePricing.isnull().sum().sort_values(ascending=False)


# In[34]:


plt.figure(figsize=(16, 16))
sns.lmplot('GrLivArea', 'SalePrice', data=HousePricing, hue='GarageCars', fit_reg=False, 
           size=10, scatter_kws={"s": 80}, palette='muted')
plt.xlabel('Living Area Size', fontsize=13)
plt.ylabel('House Price', fontsize=13)
plt.title('House Price / Living Area Size')
plt.show()


# In[35]:


plt.figure(figsize=(16, 16))
sns.lmplot('OverallQual', 'SalePrice', data=HousePricing, hue='GarageCars', fit_reg=False, 
           size=10)
plt.xlabel('Overall Quality', fontsize=13)
plt.ylabel('House Price', fontsize=13)
plt.title('Overall Quality / Price', loc='center')
plt.show()


# In[36]:


HouseByOverall = HousePricing.groupby(by='OverallQual')[['OverallQual', 'SalePrice', 'GarageCars']].count()
HouseByOverall = HouseByOverall.rename(columns={'OverallQual': 'Count'})
sns.barplot(HouseByOverall.index, 'Count', data=HouseByOverall)
plt.show()


# In[37]:


sns.barplot('OverallQual', 'SalePrice', data=HousePricing)
plt.show()

