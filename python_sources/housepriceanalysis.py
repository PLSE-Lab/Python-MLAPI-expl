#!/usr/bin/env python
# coding: utf-8

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


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


missing = data.isnull().sum()
missing = missing[missing>0]
missing.sort_values(inplace=True)
plt.figure(figsize=(15,8))
missing.plot.bar()


# In[ ]:


sns.set(rc={'figure.figsize': (12,8)})
sns.distplot(data['SalePrice'], kde=False, bins=20)


# In[ ]:


data['SalePrice'].describe()


# In[ ]:


numeric_features = data.select_dtypes(include=[np.number])
numeric_features.columns


# In[ ]:


categorical_features = data.select_dtypes(include=[np.object])
categorical_features.columns


# In[ ]:


correlation = numeric_features.corr()
print(correlation['SalePrice'].sort_values(ascending=False), '\n')


# In[ ]:


f, ax = plt.subplots(figsize=(14, 12))
plt.title('Correlation of Numeric Features with Sale Price', y=1, size=16)
sns.heatmap(correlation, square=True, vmax=0.8)


# In[ ]:


k=11
cols = correlation.nlargest(k, 'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(data[cols].values.T)
f, ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8,linewidths=0.01 ,square = True, annot=True, cmap = 'viridis',linecolor="white", xticklabels=cols.values, annot_kws = {'size':12}, yticklabels = cols.values)


# In[ ]:


sns.scatterplot(x='GarageCars', y='SalePrice', data=data)


# In[ ]:


sns.regplot(x='GarageCars', y='GarageArea', data=data, scatter=True, fit_reg = True)


# In[ ]:


fig, ((ax1,ax2), (ax3, ax4), (ax5,ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(14,10))
fig.tight_layout(pad=5.0)
sns.regplot(x='OverallQual', y='SalePrice', data=data, scatter=True, fit_reg=True, ax=ax1)
sns.regplot(x='GrLivArea', y='SalePrice', data=data, scatter=True, fit_reg=True, ax=ax2)
sns.regplot(x='GarageArea', y='SalePrice', data=data, scatter=True, fit_reg = True, ax=ax3)
sns.regplot(x='FullBath', y='SalePrice', data=data, scatter=True, fit_reg=True, ax=ax4)      
sns.regplot(x='YearBuilt', y='SalePrice', data=data, scatter=True, fit_reg =True, ax=ax5)
sns.regplot(x='WoodDeckSF', y='SalePrice', data=data, scatter=True, fit_reg = True, ax=ax6)
      


# In[ ]:


sns.boxplot(x=data['SalePrice'])


# In[ ]:


f,ax = plt.subplots(figsize = (16, 10))
fig = sns.boxplot(x='SaleType', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
xt = plt.xticks(rotation=45)


# In[ ]:


f, ax = plt.subplots(figsize = (12,8))
fig=sns.boxplot(x='OverallQual', y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)


# In[ ]:


data['SalePrice'].describe()


# In[ ]:


sns.boxplot(x=data['SalePrice'])


# In[ ]:


data.shape


# In[ ]:


first_quartile = data['SalePrice'].quantile(.25)
third_quartile = data['SalePrice'].quantile(.75)
IQR = third_quartile - first_quartile


# In[ ]:


new_boundary = third_quartile + 3*IQR


# In[ ]:


data.drop(data[data['SalePrice']>new_boundary].index, axis=0, inplace=True)


# In[ ]:


data.shape

**Remove Bad Features From Data**
1. GarageArea <-> GarageCars
2. TotalBsmtSF <-> 1stFlrSF
3. TotalRmsAbvGrd <-> GrLivArea
4. GrLivArea <-> FullBath

**Features with missing values more than 20%**
5. FireplaceQu 690/1460 = 47%
6. Fence 1179/1460 == 80%
7. Alley 1369 > 90%
8. PoolQc 1453 > 90%

**Features with Poor Correlation with the Target Feature(SalePrice) are also included in the columns to be removed**

# In[ ]:


cols_to_remove = ['BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch',
                  'PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath', 'MiscVal', 'Id','LowQualFinSF','YrSold', 'OverallCond','MSSubClass','EnclosedPorch',
                  'KitchenAbvGr','FireplaceQu','Fence','Alley','PoolQC','GarageCars','1stFlrSF','GrLivArea','FullBath','MiscFeature'
]


# In[ ]:


data.drop(cols_to_remove, axis=1, inplace=True)


# In[ ]:


data.shape


# In[ ]:


data.columns

