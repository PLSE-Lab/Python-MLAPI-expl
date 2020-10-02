#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # **1.0 Import Train and Test Data**

# In[ ]:


X_train_y = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv',header=0)
X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',header=0)
X_train_y.set_index('Id', inplace=True)
X_test.set_index('Id', inplace=True)
y_train = X_train_y['SalePrice']
X_train= X_train_y.drop('SalePrice', axis=1)


# In[ ]:


X_train.info()
print('DIM X_Train: ', X_train.shape, ' DIM X_test: ', X_test.shape )


# In[ ]:


X = pd.concat([X_train, X_test])
#X.set_index('Id', inplace=True)


# # **1.1 Finding NaN values, dealing with NaN's investigating possible Imputing**
# 
# Investigating by value_counts() on object types (which later i will switch to category). Reading Column description to find a clue. For numerical if there are most of data filled i will use either mean or median

# In[ ]:


#NULL Contain columns:
X[X.columns[X.isna().any()].tolist()].info()


# In[ ]:



X['MSZoning']=X['MSZoning'].fillna('RL')
X['LotFrontage']=X['LotFrontage'].fillna(X['LotFrontage'].median())
X.drop('GarageYrBlt',axis=1, inplace=True) # drop because Null value may be described by GarageFinish or NA, there is no option to write if there is no garaga at all
X.drop('PoolQC',axis=1, inplace=True) 
X.drop('MiscFeature',axis=1, inplace=True) 


# Alley: Type of alley access to property
# 
#        Grvl	Gravel
#        Pave	Paved
#        NA 	No alley access
#        
# Filling NaN's as NA because no access also may determine value of the house

# In[ ]:



X['Alley'].fillna('NA', inplace=True) # Fill NaN as No alley access
X['Utilities'].fillna('AllPub', inplace=True)
X['Exterior1st'].fillna('VinylSd', inplace=True)  # 1st most common
X['Exterior2nd'].fillna('MetalSd', inplace=True)  # 2nd most common
X['MasVnrType'].fillna('None', inplace=True)
X['MasVnrArea'].fillna(0, inplace=True) # Every missing value got MasVnrType "None" so filling with 0
X['BsmtQual'].fillna('TA', inplace=True)
X['BsmtCond'].fillna('TA', inplace=True)
X['BsmtExposure'].fillna('No', inplace=True)
X['BsmtFinType1'].fillna('Unf', inplace=True)
X['BsmtFinType2'].fillna('Unf', inplace=True)
X['BsmtFinSF1'].fillna(0, inplace=True)
X['BsmtFinSF2'].fillna(0, inplace=True)
X['BsmtUnfSF'].fillna(0, inplace=True)
X['TotalBsmtSF'].fillna(0, inplace=True)
X['Electrical'].fillna('SBrkr', inplace=True)
X['BsmtFullBath'].fillna(0, inplace=True)
X['BsmtHalfBath'].fillna(0, inplace=True)
X['KitchenQual'].fillna('TA', inplace=True)
X['Functional'].fillna('Typ', inplace=True)
X['FireplaceQu'].fillna('NA', inplace=True)
X['GarageType'].fillna('NA', inplace=True)
X['GarageFinish'].fillna('NA', inplace=True)
X['GarageCars'].fillna(2, inplace=True)
X['GarageArea'].fillna(X[X['GarageCars']==2]['GarageArea'].mean(), inplace=True) # fill with the mean of garages with 2 cars
X['GarageQual'].fillna('NA', inplace=True)
X['GarageCond'].fillna('NA', inplace=True)
X['Fence'].fillna('NA', inplace=True)
X['SaleType'].fillna('WD', inplace=True)


# # 1.2 Feature Selection - Category
# 
# Selecting Features which has variability inside dataset, i am removing all the columns which got one of the values distributed more than 88%

# In[ ]:


col_to_drop=[]
for col in X.select_dtypes(include='object').columns:
    if X[col].value_counts()[0]/X[col].shape[0] > 0.88:
        print('Feature: ',col,' has no valuable information because there is one value dominated over dataset with', 100*X[col].value_counts()[0]/X[col].shape[0],'% of all')
        col_to_drop.append(col)
X.drop(col_to_drop,axis=1,inplace=True)     


# # 1.3 Feature Changing - Numerical to Category
# 
# Changing types of features which are numerical to category (Months, Years etc.)****

# In[ ]:


X['MSSubClass']=X['MSSubClass'].astype('category')
X['OverallQual']=X['OverallQual'].astype('category')
X['OverallCond']=X['OverallCond'].astype('category')
X['YearBuilt']=X['YearBuilt'].astype('category')
X['YearRemodAdd']=X['YearRemodAdd'].astype('category')
X['BsmtFullBath']=X['BsmtFullBath'].astype('category')
X['BsmtHalfBath']=X['BsmtHalfBath'].astype('category')
X['FullBath']=X['FullBath'].astype('category')
X['HalfBath']=X['HalfBath'].astype('category')
X['BedroomAbvGr']=X['BedroomAbvGr'].astype('category')
X['KitchenAbvGr']=X['KitchenAbvGr'].astype('category')
X['TotRmsAbvGrd']=X['TotRmsAbvGrd'].astype('category')
X['Fireplaces']=X['Fireplaces'].astype('category')
X['GarageCars']=X['GarageCars'].astype('category')
X['MoSold']=X['MoSold'].astype('category')
X['YrSold']=X['YrSold'].astype('category')


# In[ ]:


col_to_drop=[]
for col in X.select_dtypes(include='category').columns:
    if X[col].value_counts().iloc[0]/(X[col].shape[0]) > 0.88:
        print('Feature: ',col,' has no valuable information because there is one value dominated over dataset with', 100*X[col].value_counts().iloc[0]/(X[col].shape[0]),'% of all')
        col_to_drop.append(col)
X.drop(col_to_drop,axis=1,inplace=True)


# # 1.4 Feature Changing - Object to Category
# 
# Changing types of features which are object to category

# In[ ]:


X[(X.select_dtypes(include='object').columns)]=X[(X.select_dtypes(include='object').columns)].astype('category')


# # 1.5 Feature Selection - Numerical features
# 
# Finding Features to drop and reduce dimensions, RED Axes for further investigation

# In[ ]:


X.info()


# In[ ]:


plt.figure(figsize=(30,30))
for i,col in enumerate(X.select_dtypes(exclude=['category']).columns):
    plt.subplot(4,5,i+1)
    if i in [4,9,14,15,16,17,18]:
        ax=sns.distplot(X[col],kde=False , hist=True, color='red')
    else:
        ax=sns.distplot(X[col],kde=False , hist=True)


# In[ ]:


X[['BsmtFinSF2','LowQualFinSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']].describe(percentiles=[.25, .5, .75, .9])


# All features to drop because of outliers, rest of numerical features with outliers are accepted due its true data and may be significant

# In[ ]:


X.drop(['BsmtFinSF2','LowQualFinSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal'],axis=1, inplace=True)


# Alternative drop - also because of many outliers

# In[ ]:


X[['MasVnrArea', 'WoodDeckSF', 'OpenPorchSF','2ndFlrSF', 'BsmtFinSF1']].describe(percentiles=[0.1,.25, .5, .75, .9])


# In[ ]:


X.drop(['MasVnrArea', 'WoodDeckSF', 'OpenPorchSF','2ndFlrSF', 'BsmtFinSF1'],axis=1, inplace=True)


# In[ ]:


X.info()


# # 1.6 Category Mean Encoding
# 
# after that I will try to get rid or accept the outliers

# In[ ]:


X['SalePrice']= y_train


# In[ ]:


test = pd.DataFrame()
test['SalePrice']=X['SalePrice']
test=test.sort_values(by='SalePrice').reset_index()


# # **DeSkewing**

# In[ ]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
ax=sns.distplot(X['SalePrice'],kde=False , hist=True)
ax.set_title('Dist plot of SalePrice')
plt.subplot(2,2,2)
ax=sns.distplot(np.log(X['SalePrice']),kde=False , hist=True)
ax.set_title('Dist plot of log SalePrice')

#test=sorted(test)
plt.subplot(2,2,3)
sns.scatterplot(x=test.index,y=test['SalePrice'], data=test)
plt.subplot(2,2,4)
sns.scatterplot(x=test.index,y=np.log(test['SalePrice']), data=test)
#ax=sns.distplot(X[col],kde=False , hist=True)


# 
# 
# 

# # Logarithm Mean Encoding of Categorical features

# In[ ]:


for col in X.select_dtypes(include='category').columns:
    mean_encode=(np.log(X.groupby(col)['SalePrice'].mean())-np.log(X.groupby(col)['SalePrice'].mean().min()))/(np.log(X.groupby(col)['SalePrice'].mean().max())-np.log(X.groupby(col)['SalePrice'].mean().min()))
    X[col+'_enc']= X[col].map(mean_encode).astype(float)


# In[ ]:


X.drop(X.select_dtypes(include='category').columns, axis=1, inplace=True)


# In[ ]:


X_train=X.iloc[:1460,:]


# In[ ]:


X_pred=X.iloc[1460:,:]


# In[ ]:



#for col in  X_train.columns:
plt.figure(figsize=(20,1))
#data=pd.concat([X_train[col], X_train['SalePrice']], axis=1)
#sns.boxplot( data = X[(X.select_dtypes(exclude='object').columns)],width=0.8)
sns.boxplot( x="SalePrice", data=X_train)
#plt.xscale('log')
plt.show()


# In[ ]:


pd.set_option('display.max_columns', 500)
X_train[X_train.SalePrice>500000].sort_values(by='SalePrice')


# # Correlation Matrix to figure out which sibling features to drop

# In[ ]:


#correlation matrix
corrmat = X_train.corr()
f, ax = plt.subplots(figsize=(35, 35))
sns.heatmap(corrmat, vmax=1,vmin=0.75, square=True,annot=True,cmap="YlGnBu");


# In[ ]:


X_train.drop(['1stFlrSF', 'Exterior2nd_enc', 'SaleType_enc', 'GarageCars_enc', 'TotRmsAbvGrd_enc'], axis=1, inplace=True)
X_pred.drop(['1stFlrSF', 'Exterior2nd_enc', 'SaleType_enc', 'GarageCars_enc', 'TotRmsAbvGrd_enc'], axis=1, inplace=True)


# # 2.0 Building GradientBoostingRegressor
# 
# Tutorial for Hyperparameter tuning on parameters section

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


#steps = [ ('rf', RandomForestRegressor())]#('scaler', StandardScaler()),
steps = [('scaler', StandardScaler()), ('sgb', GradientBoostingRegressor())]
pipeline = Pipeline(steps)


# In[ ]:


#parameters = {'rf__n_estimators': [100, 200, 300, 400, 500],
#'rf__max_depth': [2,4, 6, 8],
#'rf__min_samples_leaf': [0.01,0.025, 0.05, 0.1, 0.2],
#'rf__max_features': ['log2','sqrt']}
parameters = {'sgb__n_estimators': [1800],
'sgb__min_samples_split' : [27],
'sgb__max_depth': [7],
'sgb__subsample' : [0.85],
'sgb__learning_rate' : [0.005],
'sgb__min_samples_leaf': [0.015],
'sgb__warm_start' : [True],
'sgb__max_features': [8]}#range(2,30,1)}#['sqrt','log2',None]}


# **Tuning with** https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

# In[ ]:


grid_rf = GridSearchCV(estimator=pipeline,
param_grid=parameters,
cv=5,
scoring='neg_mean_squared_error',
verbose=1,
n_jobs=-1)


# In[ ]:


X=X_train.drop('SalePrice',axis=1)


# In[ ]:


y=np.log(X_train['SalePrice'])


# In[ ]:


X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


grid_rf.fit(X_train, y_train)
y_pred = grid_rf.predict(X_test)
print(grid_rf.best_params_)
print(grid_rf.best_score_)
rmse_test = MSE(y_test, y_pred)**(1/2)
print('RMSE=',rmse_test)


# In[ ]:


X_pred.drop('SalePrice',axis=1, inplace=True)


# In[ ]:


X_pred=X_pred.fillna(0)


# In[ ]:


y_pred=grid_rf.predict(X_pred)


# In[ ]:


result=pd.DataFrame()
result['Id']=X_pred.index


# In[ ]:


result['SalePrice']= np.exp(y_pred).reshape(-1,1)


# In[ ]:


result.set_index('Id', inplace=True)


# In[ ]:


result.to_csv('result_rf_sale_price_gb_tune.csv')

