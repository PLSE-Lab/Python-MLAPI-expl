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


import xgboost as xgb


# In[ ]:


train=pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv(r"/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train.tail(5)

test.tail(5)


# In[ ]:


train.columns[0:80]==test.columns[0:80]


# In[ ]:


y=np.log(train['SalePrice'])
y.head(5)


# In[ ]:


train1=train[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']]
train1.head(5)


# In[ ]:


both=pd.merge_ordered(train1,test)
both.head()


# In[ ]:


na_merge=both.fillna(value="not present")
na_merge.head(5)


# In[ ]:


combo=na_merge

combo["LotFrontage"]=combo["LotFrontage"].replace(to_replace="not present",value=0)

np.average(combo["LotFrontage"])


# In[ ]:


combo["LotFrontage"]=combo["LotFrontage"].replace(to_replace=0,value=58)


# In[ ]:


merged=both[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']]


# In[ ]:


final1=combo[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood','BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
       'RoofStyle','MasVnrType',
       'ExterQual', 'ExterCond','BsmtQual',
       'BsmtCond',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea','FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageQual','GarageCond', 'PavedDrive','3SsnPorch',
        'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold',
       'SaleCondition']]


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in final1.columns:
    final1[i] = le.fit_transform(final1[i])


# In[ ]:


final1.head(5)


# In[ ]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
X=final1[0:1460]
LR.fit(X,y)
LR.score(X,y)


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


xgb = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
xgb.fit(X,y,verbose=False)
xgb.score(X,y)


# In[ ]:


X_test=final1[1460:]


# In[ ]:


xgb_y=xgb.predict(X_test)
xgb_y


# In[ ]:


xgb_y=pd.DataFrame(data=xgb_y,index=X_test.index+1,columns=['SalePrice'])


# In[ ]:


xgb_y.head()


# In[ ]:


xgb_y['SalePrice']=np.exp(xgb_y['SalePrice'])
xgb_y.head()


# In[ ]:


xgb_y.to_csv("xgb_y.csv")


# In[ ]:


xgb_y.tail()


# In[ ]:


final=X


# In[ ]:


final['SalePrice']=y[0:1460]


# In[ ]:


cor_mat=final.corr()


# In[ ]:


pd.set_option('display.max_columns', 85)


# In[ ]:


cormat=cor_mat[51:]
cormat


# In[ ]:


final.columns


# In[ ]:


##after correlation
final2=final[['LotFrontage', 'LotArea', 'Street', 'LandContour','LandSlope',
       'Neighborhood','HouseStyle', 'OverallQual',  'YearBuilt', 'RoofStyle','ExterCond',
       'BsmtCond', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath',
       'HalfBath', 'BedroomAbvGr',  'TotRmsAbvGrd', 'Functional', 'Fireplaces',
       'GarageCond', 'PavedDrive', '3SsnPorch', 'PoolArea',
       'Fence', 'MiscFeature','MoSold', 'YrSold',
       'SaleCondition']]


# In[ ]:


test2=final1[['LotFrontage', 'LotArea', 'Street', 'LandContour','LandSlope',
       'Neighborhood','HouseStyle', 'OverallQual',  'YearBuilt', 'RoofStyle','ExterCond',
       'BsmtCond', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath',
       'HalfBath', 'BedroomAbvGr',  'TotRmsAbvGrd', 'Functional', 'Fireplaces',
       'GarageCond', 'PavedDrive', '3SsnPorch', 'PoolArea',
       'Fence', 'MiscFeature','MoSold', 'YrSold',
       'SaleCondition']]


# In[ ]:


test2.info()


# In[ ]:


X2=final2[0:1460]


# In[ ]:


LR.fit(X2,y)


# In[ ]:


LR.score(X2,y)


# In[ ]:


xgb.fit(X2,y,verbose=False)
xgb.score(X2,y)


# In[ ]:


X_test2=test2[1460:]


# In[ ]:


xgb_y2=xgb.predict(X_test2)
xgb_y2


# In[ ]:


xgb_y2=pd.DataFrame(data=xgb_y2,index=X_test2.index+1,columns=['SalePrice'])


# In[ ]:


xgb_y2['SalePrice']=np.exp(xgb_y2['SalePrice'])
xgb_y2.head()


# In[ ]:


xgb_y2.to_csv("xgb_y2.csv")


# In[ ]:




