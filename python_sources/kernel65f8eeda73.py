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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


df.head()


# In[ ]:


df['MSZoning'].value_counts()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())


# In[ ]:



df.drop(['Alley'],axis=1,inplace=True)


# In[ ]:


df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])


# In[ ]:


df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])


# In[ ]:


df.drop(['GarageYrBlt'],axis=1,inplace=True)


# In[ ]:


df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])


# In[ ]:


df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:



df.drop(['Id'],axis=1,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[ ]:


df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')


# In[ ]:


df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:



columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


# In[ ]:



len(columns)


# In[ ]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[ ]:


main_df=df.copy()


# In[ ]:


test_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:



test_df.shape


# In[ ]:


test_df.head()


# In[ ]:


final_df=pd.concat([df,test_df],axis=0)


# In[ ]:


final_df['SalePrice']


# In[ ]:


final_df.shape


# In[ ]:


final_df=category_onehot_multcols(columns)


# In[ ]:


final_df.shape


# In[ ]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[ ]:


final_df.shape


# In[ ]:


final_df


# In[ ]:



df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]


# In[ ]:


df_Train.head()


# In[ ]:


df_Test.head()


# In[ ]:


df_Train.shape


# In[ ]:



df_Test.drop(['SalePrice'],axis=1,inplace=True)


# In[ ]:


X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']


# In[ ]:


import xgboost
classifier=xgboost.XGBRegressor()


# In[ ]:


import xgboost
regressor=xgboost.XGBRegressor()


# In[ ]:


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]


# In[ ]:


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[ ]:


random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)


# In[ ]:





# In[ ]:





# In[ ]:




