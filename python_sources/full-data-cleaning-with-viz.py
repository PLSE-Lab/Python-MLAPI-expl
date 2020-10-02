#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


###importing necesary libraries...
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


null_columns=train.columns[train.isnull().any()]
train[null_columns].isnull().sum()


# In[ ]:


train=train.drop(columns=["PoolQC","Fence","MiscFeature","Alley","FireplaceQu"])
test=test.drop(columns=["PoolQC","Fence","MiscFeature","Alley","FireplaceQu"])
train.head()


# In[ ]:


null_columns=train.columns[train.isnull().any()]
train[null_columns].isnull().sum()


# In[ ]:


train[null_columns].head()


# In[ ]:


for col in null_columns:
    print(col,train[col].unique())


# In[ ]:


train["MasVnrArea"].fillna(train.MasVnrArea.median(),inplace=True)
train["GarageYrBlt"].fillna(train["YearBuilt"],inplace=True)
test["MasVnrArea"].fillna(train.MasVnrArea.median(),inplace=True)
test["GarageYrBlt"].fillna(test["YearBuilt"],inplace=True)


# In[ ]:


null_columns=train.columns[train.isnull().any()]
train[null_columns].isnull().sum()


# In[ ]:


train=train.drop(columns=["LotFrontage"])
test=test.drop(columns=["LotFrontage"])
null_columns=train.columns[train.isnull().any()]
train[null_columns].isnull().sum()


# In[ ]:


train["BsmtQual"].fillna('Fa',inplace=True)
train["GarageFinish"].fillna('Fa',inplace=True)
train["GarageQual"].fillna('Fa',inplace=True)
train["GarageCond"].fillna('Fa',inplace=True)
train["BsmtCond"].fillna('Fa',inplace=True)
train["MasVnrType"].fillna('None',inplace=True)
train["BsmtCond"].fillna('No',inplace=True)
train["BsmtExposure"].fillna('No',inplace=True)
train["BsmtFinType1"].fillna('Unf',inplace=True)
train["BsmtFinType2"].fillna('Unf',inplace=True)
train["Electrical"].fillna('Mix',inplace=True)
train["GarageType"].fillna('BuiltIn',inplace=True)
#test
test["BsmtQual"].fillna('Fa',inplace=True)
test["GarageFinish"].fillna('Fa',inplace=True)
test["GarageQual"].fillna('Fa',inplace=True)
test["GarageCond"].fillna('Fa',inplace=True)
test["BsmtCond"].fillna('Fa',inplace=True)
test["MasVnrType"].fillna('None',inplace=True)
test["BsmtCond"].fillna('No',inplace=True)
test["BsmtExposure"].fillna('No',inplace=True)
test["BsmtFinType1"].fillna('Unf',inplace=True)
test["BsmtFinType2"].fillna('Unf',inplace=True)
test["Electrical"].fillna('Mix',inplace=True)
test["GarageType"].fillna('BuiltIn',inplace=True)


# In[ ]:


plt.figure(figsize=(30,20))
num_feat=train.select_dtypes(include = 'number').columns
sns.heatmap(train[num_feat].corr(),cmap="YlGnBu",linewidths=0.5,annot=True)


# In[ ]:


for i in num_feat:
    if train[i].corr(train["SalePrice"])>0.3:
        print(i,"-","SalesPrice:",train[i].corr(train["SalePrice"]))


# In[ ]:


plt.scatter(train["OverallQual"],train["SalePrice"])
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")


# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(121)
sns.distplot(train["OverallQual"], kde=False, bins=20)
plt.subplot(122)
sns.distplot(train["OverallQual"], hist=False, bins=20)
sns.relplot(x="OverallQual", y="SalePrice", kind="line", data=train)


# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(121)
sns.distplot(train["GarageCars"], kde=False, bins=20)
plt.subplot(122)
sns.distplot(train["GarageCars"], hist=False, bins=20)
sns.relplot(x="GarageCars", y="SalePrice", kind="line", data=train)


# In[ ]:


sns.relplot(x="GarageCars", y="GarageArea", kind="line", data=train)


# In[ ]:


sns.relplot(x="GrLivArea", y="SalePrice", kind="line", data=train)


# In[ ]:


null_columns=test.columns[test.isnull().any()]
test[null_columns].isnull().sum()


# In[ ]:



num_feat=test.select_dtypes(include = 'number').columns
print(num_feat)


# In[ ]:


test["BsmtFinSF1"].fillna(train.BsmtFinSF1.median(),inplace=True)
test["BsmtFinSF2"].fillna(train.BsmtFinSF2.median(),inplace=True)
test["BsmtUnfSF"].fillna(train.BsmtUnfSF.median(),inplace=True)
test["TotalBsmtSF"].fillna(train.TotalBsmtSF.median(),inplace=True)
test["BsmtFullBath"].fillna(train.BsmtFullBath.median(),inplace=True)
test["BsmtHalfBath"].fillna(train.BsmtHalfBath.median(),inplace=True)
test["GarageCars"].fillna(train.GarageCars.median(),inplace=True)
test["GarageArea"].fillna(train.GarageArea.median(),inplace=True)


# In[ ]:


null_columns=test.columns[test.isnull().any()]
for i in null_columns:
    if i in num_feat:
        print(i)
test[null_columns].isnull().sum()
y=train['SalePrice']
train=train.drop(columns=['SalePrice'])
num_feat=train.select_dtypes(include = 'number').columns


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
print(-1 * cross_val_score(RandomForestRegressor(50),train[num_feat],y,scoring = 'neg_mean_absolute_error').mean())
regr= RandomForestRegressor(max_depth=50,random_state=0)
regr.fit(train[num_feat], y)


# In[ ]:


ans=regr.predict(test[num_feat])


# In[ ]:


len(ans)


# In[ ]:


index=[]
for i in range(1461,2920):
    index.append(i)
print(index)


# In[ ]:


index= np.array(index)
dataset = pd.DataFrame({'Id': index, 'SalePrice':ans}, columns=['Id', 'SalePrice'])


# In[ ]:


print(dataset)
print(len(dataset))


# In[ ]:


dataset.to_csv('submission.csv')


# In[ ]:



sns.jointplot(train['YearBuilt'],train['Fireplaces'],kind='kde',color='r')


# In[ ]:


f, ax = plt.subplots(figsize=(20, 8))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x=train['GrLivArea'],y=y)


# In[ ]:



sns.jointplot(train['GrLivArea'],y,kind='kde',color='g',height=7)


# In[ ]:




