#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df.head()


# In[ ]:


Df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
Df.head()


# In[ ]:


Df.shape


# In[ ]:


Df.ndim


# In[ ]:


Df.columns


# In[ ]:


Df.describe


# In[ ]:


Df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.dtypes


# In[ ]:


df.ndim


# In[ ]:


df.describe


# In[ ]:


df.isnull().sum()


# In[ ]:


df.isna().sum()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df.info()


# FILL MISSING VALUES

# In[ ]:


df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[ ]:


Df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[ ]:


Df['MSZoning'].value_counts()


# In[ ]:


Df['MSZoning'].fillna(df['MSZoning'].mode()[0])


# In[ ]:


df['MSZoning'].value_counts()


# In[ ]:


df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])


# In[ ]:


Df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])


# In[ ]:


df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])


# In[ ]:


Df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])


# In[ ]:


df['GarageType'].fillna(df['GarageType'].mode()[0])


# In[ ]:


Df['GarageType'].fillna(df['GarageType'].mode()[0])


# In[ ]:


df['GarageFinish'].fillna(df['GarageType'].mode()[0])


# In[ ]:


Df['GarageFinish'].fillna(df['GarageType'].mode()[0])


# In[ ]:


df['GarageQual'].fillna(df['GarageType'].mode()[0])


# In[ ]:


Df['GarageQual'].fillna(df['GarageType'].mode()[0])


# In[ ]:


df['GarageCond'].fillna(df['GarageType'].mode()[0])


# In[ ]:


df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])


# In[ ]:


Df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])


# In[ ]:


df['MasVnrArea'].fillna(df['MasVnrType'].mode()[0])


# In[ ]:


Df['MasVnrArea'].fillna(df['MasVnrType'].mode()[0])


# In[ ]:


df.drop(['Alley'],axis=1,inplace=True)


# In[ ]:


Df.drop(['Alley'],axis=1,inplace=True)


# In[ ]:


df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


Df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[ ]:


Df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[ ]:


df.drop(['Id'],axis=1,inplace=True)


# In[ ]:


Df.drop(['Id'],axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


Df.shape


# In[ ]:


Df.isnull().sum()


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[ ]:


sns.heatmap(Df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[ ]:


df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='BuGn')


# In[ ]:


df.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'],axis=1,inplace=True)


# In[ ]:


Df.drop(['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'],axis=1,inplace=True)


# In[ ]:


sns.heatmap(Df.isnull(),yticklabels=False,cbar=False,cmap='BuGn')


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


Df.dropna(inplace=True)


# In[ ]:


df.shape


# In[ ]:


Df.shape


# In[ ]:


df.head()


# In[ ]:


Df.head()


# In[ ]:


df.columns


# In[ ]:


Df.columns


# In[ ]:


def Category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df(fields),drop_first=True)
        
        final_df.drop([feilds],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
        
    df_final=pd.concat([final_df,df_final],axis=1)
    return df_final
        
        


# In[ ]:


Df.to_csv('formulatedtest.csv',index=False)


# In[ ]:


main_df=df.copy()


# # Combine test data

# In[ ]:


test_df=pd.read_csv("formulatedtest.csv")


# In[ ]:


test_df.shape


# In[ ]:


test_df.head()


# In[ ]:


final_df=pd.concat([df,test_df],axis=0)


# In[ ]:


final_df.shape


# In[ ]:


final_df.head()


# In[ ]:


final_df.loc[:,~final_df.columns.duplicated()]


# In[ ]:


df_train=final_df.iloc[:572,:]


# In[ ]:


df_test=final_df.iloc[:572,:]


# In[ ]:


df_test.drop(['SalePrice'],axis=1,inplace=True)


# In[ ]:


df_test.shape


# In[ ]:


x_train=df_train.drop(['SalePrice'],axis=1)


# In[ ]:


y_train=df_train['SalePrice']


# In[ ]:


import xgboost
classifier=xgboost.XGBRegressor()
classifier.fit(x_train,y_train)


# In[ ]:


import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier,open(filename,'wb'))


# In[ ]:


y_pred=classifier.predict(df_test)


# In[ ]:


y_pred


# # Create sample submisson file and submit

# In[ ]:


pred=pd.DataFrame(y_pred)


# In[ ]:


sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# In[ ]:


datasets=pd.concat([sub_df['Id'],pred],axis=1)


# In[ ]:


datasets.to_csv('sample_submission.csv',index=False)

