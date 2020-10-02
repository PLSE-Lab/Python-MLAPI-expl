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


train1=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test1=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train1.shape


# In[ ]:


test1.shape


# In[ ]:


train1.head()


# In[ ]:


train1.info()


# Numerical features:
# * MSSubclass
# * LotArea
# * BsmtFinSF1
# * BsmtFinSF2
# * BsmtUnfSF
# * TotalBsmtSF
# * 1stFlrSF
# * 2ndFlrSF
# * LowQualFinSF
# * GrLivArea
# * GarageArea
# * WoodDeckSF
# * OpenPorchSF
# * EnclosedPorch
# * 3SsnPorch
# * PoolArea
# * MiscVal

# In[ ]:


num=train1[['MSSubClass','LotArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF' ,'1stFlrSF','2ndFlrSF',
              'LowQualFinSF' ,'GrLivArea', 'GarageArea','WoodDeckSF' ,'OpenPorchSF','EnclosedPorch',
              '3SsnPorch','PoolArea','MiscVal','LotFrontage','MasVnrArea','ScreenPorch','SalePrice']]


# In[ ]:


num.head()


# In[ ]:


num.describe().transpose()


# In[ ]:


num['TotBsmtFin']=num['BsmtFinSF1']+num['BsmtFinSF2']


# In[ ]:


num=num.drop(['BsmtFinSF1','BsmtFinSF2'],axis=1)


# In[ ]:


num['LowQualFinSF'].value_counts().sort_index()
#this turned out to be a categorical variable


# In[ ]:


num.isnull().sum()


# In[ ]:


num['MasVnrArea'].fillna(103.685262	,inplace=True)
num['LotFrontage'].fillna(70.049958	,inplace=True)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(14,14))
sns.heatmap(num.corr(),annot=True,cmap='coolwarm')


# In[ ]:


sns.jointplot(x='TotalBsmtSF',y='SalePrice',data=num)


# Got an outliar!

# In[ ]:


train1[train1['TotalBsmtSF']>6000]
#This is an outliar


# In[ ]:


train1=train1.drop(1298)


# In[ ]:


sns.jointplot(x='GrLivArea',y='SalePrice',data=num)


# One more Outliar!

# In[ ]:


train1[train1['GrLivArea']>4000]['TotalBsmtSF']
#523rd row is an outliar


# In[ ]:


train1=train1.drop(523)


# In[ ]:


sns.jointplot(x='MasVnrArea',y='SalePrice',data=num)


# Moderatly strong correlation bewteen the two variables can be seen.

# In[ ]:


train1[train1['MasVnrArea']>1500]['GrLivArea']
#not an outliar


# In[ ]:


train1[(train1.MasVnrArea ==0) & (train1.SalePrice >700000)]['GrLivArea']
train1[(train1.MasVnrArea ==0) & (train1.SalePrice >700000)]['TotalBsmtSF']
#not an outliar


# In[ ]:


sns.jointplot(x='GarageArea',y='SalePrice',data=num)


# In[ ]:


train1[train1['GarageArea']>1200]['GrLivArea']
train1[train1['GarageArea']>1200]['MasVnrArea']
train1[train1['GarageArea']>1200]['TotalBsmtSF']


# In[ ]:


train1=train1.drop(1061)


#  Categorical features:
# * MSZoning
# * Street
# * LotShape
# * LandContour
# * Utilities
# * LotConfig
# * LandSlope
# * Neighborhood
# * Condition1
# * Condition2
# * BldgType
# * HouseStyle
# * OverallQual
# * OverallCond
# * YearBuilt
# * YearRemodAdd
# * RoofStyle
# * RoofMatl
# * Exterior1st
# * Exterior2nd
# * MasVnrType
# * ExterQual
# * ExterCond
# * Foundation
# * BsmtQual
# * BsmtCond
# * BsmtExposure
# * BsmtFinType1
# * BsmtFinType2
# * Heating
# * HeatingQC
# * Electrical
# * BsmtFullBath
# * BsmtHalfBath
# * HalfBath
# * FullBath
# * BedroomAbvGr
# * KitchenAbvGr
# * KitchenQual
# * TotRmsAbvGrd
# * Functional
# * Fireplaces
# * GarageType
# * GarageYrBlt
# * GarageFinish
# * GarageCars

# In[ ]:


cat=train1.drop(['MSSubClass','LotArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', 'TotalBsmtSF' ,'1stFlrSF','2ndFlrSF',
              'LowQualFinSF' ,'GrLivArea', 'GarageArea','WoodDeckSF' ,'OpenPorchSF','EnclosedPorch',
              '3SsnPorch','PoolArea','MiscVal','CentralAir','LotFrontage','MasVnrArea','ScreenPorch'],axis=1)


# In[ ]:


cat.head()


# In[ ]:


cat.info()


# In[ ]:


cat.isnull().sum().sort_values(ascending=False).head(15)


# In[ ]:


cat['GarageFinish'].value_counts()


# In[ ]:


cat['GarageFinish'].fillna('Unf',inplace=True)


# In[ ]:


#GarageQual is same as GarageCond
cat['GarageQual'].value_counts()


# In[ ]:


cat['GarageQual'].fillna('TA',inplace=True)
cat['GarageCond'].fillna('TA',inplace=True)


# In[ ]:


cat['GarageYrBlt'].value_counts()


# In[ ]:


cat['GarageYrBlt'].fillna(2005.0,inplace=True)


# In[ ]:


cat=cat.astype({'GarageYrBlt':'int64'})


# In[ ]:


cat['GarageYrBlt'].dtype


# In[ ]:


cat['GarageType'].value_counts()


# In[ ]:


cat['GarageType'].fillna('Attchd',inplace=True)


# In[ ]:


cat['BsmtExposure'].value_counts()


# In[ ]:


cat['BsmtExposure'].fillna('No',inplace=True)


# In[ ]:


cat['BsmtFinType1'].value_counts()
cat['BsmtFinType1'].fillna('Unf',inplace=True)
cat['BsmtFinType2'].value_counts()
cat['BsmtFinType2'].fillna('Unf',inplace=True)


# In[ ]:


cat['BsmtQual'].value_counts()
cat['BsmtQual'].fillna('TA',inplace=True)
cat['BsmtCond'].value_counts()
cat['BsmtQual'].fillna('TA',inplace=True)


# In[ ]:


cat['MasVnrType'].value_counts()
cat['MasVnrType'].fillna('None',inplace=True)


# In[ ]:


cat['Electrical'].value_counts()
cat['Electrical'].fillna('SBrkr',inplace=True)


# In[ ]:


cat['Year_diff']=cat['YearRemodAdd']-cat['YearBuilt']


# In[ ]:


plt.figure(figsize=(14,14))
sns.heatmap(cat.corr(),annot=True,cmap='coolwarm')


# In[ ]:


sns.boxplot(x='OverallQual',y='SalePrice',data=cat)


# In[ ]:


sns.boxplot(x='YearBuilt',y='SalePrice',data=cat)


# In[ ]:


sns.boxplot(x='FullBath',y='SalePrice',data=cat)


# In[ ]:


sns.boxplot(x='Fireplaces',y='SalePrice',data=cat)


# Binary features:
# * CentralAir

# In[ ]:


bi=train1[['CentralAir','SalePrice']]
#bi.describe()


# In[ ]:


bi.isnull().sum()


# In[ ]:


sns.boxplot(x='CentralAir',y='SalePrice',data=bi)


# In[ ]:


from sklearn.preprocessing  import LabelEncoder
le=LabelEncoder()
bi['x']=le.fit_transform(bi['CentralAir'])


# In[ ]:


bi.corr()


# In[ ]:


s=train1.isnull().sum().sort_values(ascending=False).head(20)


# In[ ]:


percent=s/1460*100
percent
#represents the percentage of missing data, if more than 15% data is missing just drop the column


# In[ ]:


train1.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','Id'],axis=1,inplace=True)


# In[ ]:


train=train1[['OverallQual','YearBuilt', 'YearRemodAdd','TotalBsmtSF','GrLivArea','FullBath','TotRmsAbvGrd', 'GarageArea',
              'Fireplaces','MasVnrArea','SalePrice']]
train.head()


# In[ ]:


train.isnull().sum()


# In[ ]:


train['MasVnrArea'].fillna(104,inplace=True)


# In[ ]:


train.info()


# In[ ]:


X=train.drop(['SalePrice'],axis=1)
train['SalePrice']=np.log1p(train['SalePrice'])
y=train["SalePrice"]


# In[ ]:


X.std()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10)


# In[ ]:


from sklearn.model_selection import cross_val_score
def scores(model):
    scor=np.sqrt(-cross_val_score(model,X_train,y_train,scoring='neg_mean_squared_error',cv=5))
    return scor


# In[ ]:


import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.2,learning_rate=0.06,max_depth=3,n_estimators=1150)
model_xgb.fit(X_train,y_train)


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
para={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,40,50,70,100]}
l=Lasso()
lasso=GridSearchCV(l,para,scoring='neg_mean_squared_error',cv=5)
lasso.fit(X_train,y_train)


# In[ ]:


lasso.best_params_


# In[ ]:


from sklearn.preprocessing import RobustScaler


# In[ ]:


from sklearn.pipeline import make_pipeline
model_lasso=make_pipeline(RobustScaler(),Lasso(alpha=0.001))


# In[ ]:


lasso_score=scores(model_lasso).mean()
xbg_score=scores(model_xgb).mean()
svr_score=scores(svr).mean()


# In[ ]:


print(lasso_score)


# In[ ]:


print(xbg_score)


# In[ ]:


print(svr_score)


# In[ ]:


from sklearn.svm import SVR
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))


# In[ ]:


from sklearn.ensemble import StackingRegressor
stack=StackingRegressor(estimators=(model_xgb, svr, model_lasso),
                                final_estimator=model_xgb)


# In[ ]:


stack.fit(np.array(X_train),np.array(y_train))


# In[ ]:




