#!/usr/bin/env python
# coding: utf-8

# ### This notebook is a part of a learning process, adapted form the [notebook of Pedro Marcelino](http://https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_raw_data = pd.read_csv('../input/train.csv')
test_raw_data = pd.read_csv('../input/test.csv')


# In[ ]:


train_raw_data.shape


# In[ ]:


test_raw_data.shape


# ## Analyzing relationship between variables

# In[ ]:


corr_mat = train_raw_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat, vmax=0.8, square=True)


# In[ ]:


k = 10
cols = corr_mat.nlargest(k, ['SalePrice'])['SalePrice'].index
cm = np.corrcoef(train_raw_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=cols.values, annot_kws={'size': 10}, 
                 xticklabels=cols.values)
plt.show()


# In[ ]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_raw_data[cols], size=2.5)
plt.show()


# In[ ]:


n_train = train_raw_data.shape[0]
n_test = test_raw_data.shape[0]
all_data = pd.concat((train_raw_data, test_raw_data), sort=True).reset_index(drop=True)
all_data.drop(['SalePrice'], inplace=True, axis=1)


# In[ ]:


all_data.shape


# ## Dealing with missing data

# In[ ]:


def missing_data_stats():
    total = all_data.isnull().sum().sort_values(ascending=False)
    percent = (all_data.isnull().sum() / all_data.shape[0]).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(40))


# In[ ]:


missing_data_stats()


# In[ ]:


fill_na_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass']

for col in fill_na_cols:
    all_data[col] = all_data[col].fillna("None")


# In[ ]:


fill_zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 
                  'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

for col in fill_zero_cols:
    all_data[col] = all_data[col].fillna(0)


# In[ ]:


all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[ ]:


fill_mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']

for col in fill_mode_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])


# In[ ]:


all_data = all_data.drop(['Utilities'], axis=1)


# In[ ]:


all_data['Functional'] = all_data['Functional'].fillna('Typ')


# In[ ]:


missing_data_stats()


# In[ ]:


train_data = pd.concat((all_data[:n_train], train_raw_data['SalePrice']), axis=1).reset_index(drop=True)
train_data.shape


# In[ ]:


test_data = all_data[n_train:]
test_data.shape


# ## Dealing with outliers

# In[ ]:


sale_price_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:, np.newaxis])
low_range = np.sort(sale_price_scaled, axis=0)[:10]
high_range = np.sort(sale_price_scaled, axis=0)[-10:]
print('low range of the distribution')
print(low_range)
print('high range of the distribution')
print(high_range)


# In[ ]:


var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x = var, y = 'SalePrice', ylim = (0, 800000))


# In[ ]:


train_data.sort_values(by = 'GrLivArea', ascending=False)[:2]


# In[ ]:


train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
train_data = train_data.drop(train_data[train_data['Id'] == 524].index)

all_data = all_data.drop(all_data[all_data['Id'] == 1299].index)
all_data = all_data.drop(all_data[all_data['Id'] == 524].index)


# In[ ]:


var = 'TotalBsmtSF'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# ## Testing normality assumptions

# In[ ]:


def check_normal(dist):
    sns.distplot(dist, fit=norm)
    fig = plt.figure()
    res = stats.probplot(dist, plot=plt)


# In[ ]:


check_normal(train_data['SalePrice'])


# In[ ]:


train_data['SalePrice'] = np.log1p(train_data['SalePrice'])


# In[ ]:


check_normal(train_data['SalePrice'])


# In[ ]:


check_normal(train_data['GrLivArea'])


# In[ ]:


train_data['GrLivArea'] = np.log1p(train_data['GrLivArea'])
all_data['GrLivArea'] = np.log1p(all_data['GrLivArea'])


# In[ ]:


check_normal(train_data['GrLivArea'])


# In[ ]:


check_normal(train_data['TotalBsmtSF'])


# In[ ]:


train_data['TotalBsmtSF'] = np.log1p(train_data['TotalBsmtSF'])
all_data['TotalBsmtSF'] = np.log1p(all_data['TotalBsmtSF'])


# In[ ]:


check_normal(train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'])


# In[ ]:


plt.scatter(train_data['GrLivArea'], train_data['SalePrice'])


# In[ ]:


plt.scatter(train_data[train_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], train_data[train_data['TotalBsmtSF'] > 0]['SalePrice'])

