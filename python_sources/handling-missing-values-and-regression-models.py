#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:yellowgreen">Hey Folks, In case you like my little effort here please do <span style="color:green">UPVOTE</span> the kernel. Have an awesome day :) 

# # **<h1 style='color:red'> Variable Information :**

# * SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * LotFrontage: Linear feet of street connected to property
# * LotArea: Lot size in square feet
# * Street: Type of road access
# * Alley: Type of alley access
# * LotShape: General shape of property
# * LandContour: Flatness of the property
# * Utilities: Type of utilities available
# * LotConfig: Lot configuration
# * LandSlope: Slope of property
# * Neighborhood: Physical locations within Ames city limits
# * Condition1: Proximity to main road or railroad
# * Condition2: Proximity to main road or railroad (if a second is present)
# * BldgType: Type of dwelling
# * HouseStyle: Style of dwelling
# * OverallQual: Overall material and finish quality
# * OverallCond: Overall condition rating
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date
# * RoofStyle: Type of roof
# * RoofMatl: Roof material
# * Exterior1st: Exterior covering on house
# * Exterior2nd: Exterior covering on house (if more than one material)
# * MasVnrType: Masonry veneer type
# * MasVnrArea: Masonry veneer area in square feet
# * ExterQual: Exterior material quality
# * ExterCond: Present condition of the material on the exterior
# * Foundation: Type of foundation
# * BsmtQual: Height of the basement
# * BsmtCond: General condition of the basement
# * BsmtExposure: Walkout or garden level basement walls
# * BsmtFinType1: Quality of basement finished area
# * BsmtFinSF1: Type 1 finished square feet
# * BsmtFinType2: Quality of second finished area (if present)
# * BsmtFinSF2: Type 2 finished square feet
# * BsmtUnfSF: Unfinished square feet of basement area
# * TotalBsmtSF: Total square feet of basement area
# * Heating: Type of heating
# * HeatingQC: Heating quality and condition
# * CentralAir: Central air conditioning
# * Electrical: Electrical system
# * 1stFlrSF: First Floor square feet
# * 2ndFlrSF: Second floor square feet
# * LowQualFinSF: Low quality finished square feet (all floors)
# * GrLivArea: Above grade (ground) living area square feet
# * BsmtFullBath: Basement full bathrooms
# * BsmtHalfBath: Basement half bathrooms
# * FullBath: Full bathrooms above grade
# * HalfBath: Half baths above grade
# * Bedroom: Number of bedrooms above basement level
# * Kitchen: Number of kitchens
# * KitchenQual: Kitchen quality
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * Functional: Home functionality rating
# * Fireplaces: Number of fireplaces
# * FireplaceQu: Fireplace quality
# * GarageType: Garage location
# * GarageYrBlt: Year garage was built
# * GarageFinish: Interior finish of the garage
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * GarageQual: Garage quality
# * GarageCond: Garage condition
# * PavedDrive: Paved driveway
# * WoodDeckSF: Wood deck area in square feet
# * OpenPorchSF: Open porch area in square feet
# * EnclosedPorch: Enclosed porch area in square feet
# * 3SsnPorch: Three season porch area in square feet
# * ScreenPorch: Screen porch area in square feet
# * PoolArea: Pool area in square feet
# * PoolQC: Pool quality
# * Fence: Fence quality
# * MiscFeature: Miscellaneous feature not covered in other categories
# * MiscVal: Money Value of miscellaneous feature
# * MoSold: Month Sold
# * YrSold: Year Sold
# * SaleType: Type of sale
# * SaleCondition: Condition of sale

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


# # <h1 style='color:red'>Importing libraries :

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy import stats

from sklearn.linear_model import LinearRegression,Lasso,ElasticNet,Ridge
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor,VotingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# # **<h1 style='color:red'>Loading the dataset :**

# In[ ]:


train = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


train.isnull().any().describe()


# In[ ]:


test.isnull().any().describe()


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


Y = train['SalePrice']
train = train.drop('SalePrice',axis=1)
data = pd.concat([train,test],axis=0)
data = data.reset_index(drop=True)
data.shape


# # **<h1 style='color:red'>Duplicate Values :**

# In[ ]:


print("Number of duplicate values in train set : ",train.duplicated().sum())
print("Number of duplicate values in test set : ",test.duplicated().sum())


# # **<h1 style="color:red">Null Values :**

# In[ ]:


data_null = (data.isnull().sum() / len(data)) * 100
print(data_null)
data_null = data_null.drop(data_null[data_null == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :data_null})
ms = (missing_data.head(30)).style.background_gradient(low=0,high=1,axis=0,cmap='Oranges')
ms


# **<h1 style="color:cyan">MSZoning :**

# In[ ]:


data['MSZoning'].unique()


# In[ ]:


data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])


# **<h1 style="color:cyan">Lot Frontage :**

# In[ ]:


data['LotFrontage'].unique()


# In[ ]:


data['LotFrontage'].median()


# In[ ]:


data['LotFrontage'] = data['LotFrontage'].fillna(68)


# **<h1 style="color:cyan">Alley :**

# In[ ]:


data['Alley'].unique()


# In[ ]:


data["Alley"] = data["Alley"].fillna("NA")


# **<h1 style="color:cyan">Utilities :**

# In[ ]:


data['Utilities'].unique()


# In[ ]:


data['Utilities'].isnull().sum()


# In[ ]:


data['Utilities'] = data['Utilities'].fillna('AllPub')


# **<h1 style="color:cyan">Exterior1st and Exterior2nd:**

# In[ ]:


print(data['Exterior1st'].unique())
print(data['Exterior2nd'].unique())


# In[ ]:


data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])


# **<h1 style="color:cyan">MasVnrType and MasVnrArea:**

# In[ ]:


print(data['MasVnrType'].unique())
print(data['MasVnrArea'].unique())


# In[ ]:


data["MasVnrType"] = data["MasVnrType"].fillna("NA")
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)


# **<h1 style="color:cyan">BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : **

# <p style="color:magenta">These features are basement related features which have categorical values so i am going to replace the missing or null values with 'NA' which means thate there is no basement present in the houses. 

# In[ ]:


print(data['BsmtQual'].unique())
print(data['BsmtCond'].unique())
print(data['BsmtExposure'].unique())
print(data['BsmtFinType1'].unique())
print(data['BsmtFinType2'].unique())


# In[ ]:


data['BsmtQual'] = data['BsmtQual'].fillna('NA')
data['BsmtCond'] = data['BsmtCond'].fillna('NA')
data['BsmtFinType1'] = data['BsmtFinType1'].fillna('NA')
data['BsmtExposure'] = data['BsmtExposure'].fillna('NA')
data['BsmtFinType2'] = data['BsmtFinType2'].fillna('NA')


# **<h1 style="color:cyan">BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath :**

# <p style="color:magenta">These features have numerical values unlike the above Bsmt features which had categorical values i will replace the values with '0' which indicate that there is no basement present in the houses.

# In[ ]:


data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0)
data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0)
data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(0)
data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0)
data['BsmtFullBath'] = data['BsmtFullBath'].fillna(0)
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(0)


# **<h1 style="color:cyan">KitchenQual :**

# In[ ]:


data['KitchenQual'].unique()


# In[ ]:


data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])


# **<h1 style="color:cyan">Electrical :**

# In[ ]:


data['Electrical'].unique()


# In[ ]:


data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])


#  **<h1 style="color:cyan">Functional :**

# In[ ]:


data['Functional'].unique()


# In[ ]:


data['Functional'] = data['Functional'].fillna(data['Functional'].mode()[0])


# **<h1 style="color:cyan">GarageYrBlt, GarageArea and GarageCars :**

# In[ ]:


data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
data['GarageArea'] = data['GarageArea'].fillna(0)
data['GarageCars'] = data['GarageCars'].fillna(0)


# **<h1 style="color:cyan">GarageType, GarageFinish, GarageQual and GarageCond :**

# In[ ]:


data['GarageType'] = data['GarageType'].fillna('NA')
data['GarageFinish'] = data['GarageFinish'].fillna('NA')
data['GarageQual'] = data['GarageQual'].fillna('NA')
data['GarageCond'] = data['GarageCond'].fillna('NA')


# **<h1 style="color:cyan">FireplaceQu :**

# In[ ]:


data['FireplaceQu'].unique()


# In[ ]:


data["FireplaceQu"] = data["FireplaceQu"].fillna('NA')


# **<h1 style="color:cyan">Fence  :**

# In[ ]:


data['Fence'].unique()


# In[ ]:


data["Fence"] = data["Fence"].fillna('NA')


# **<h1 style="color:cyan">MiscFeature  :**

# In[ ]:


data['MiscFeature'].unique()


# In[ ]:


data["MiscFeature"] = data["MiscFeature"].fillna("NA")


# **<h1 style="color:cyan">PoolQC :**

# In[ ]:


data['PoolQC'].unique()


# In[ ]:


data["PoolQC"] = data["PoolQC"].fillna("NA")


# **<h1 style="color:cyan">SaleType :**

# In[ ]:


data['SaleType'].unique()


# In[ ]:


data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])


# **<h1 style="color:cyan">MSSubClass :**

# In[ ]:


data['MSSubClass'].unique()


# In[ ]:


data['MSSubClass'] = data['MSSubClass'].fillna("NA")


# <p style="color:magenta">All the missing values which were numerical are filled with 0's and the values which were categorical are filled with 'NA' both of these mean that the feature does not exist for that type of datapoint.No datapoint has been removed from the data.

# **<h1 style="color:cyan">After removing all null values from the dataset :**

# In[ ]:


data_null = (data.isnull().sum() / len(data)) * 100
print(data_null)


# **<p style="color:magenta">There is no missing data in the dataset left.**

# In[ ]:


train = data[:train.shape[0]]
test = data[train.shape[0]:]
train['SalePrice'] = Y


# # **<h1 style="color:red">Outliers :**

# In[ ]:


print(train.columns.values)
print(train.shape)
print(test.shape)


# In[ ]:


fig = px.scatter(train,x='LotArea',y='SalePrice',color='SalePrice',size='SalePrice')
fig.show()


# In[ ]:


train = train[train['LotArea']<100000]
print(train.shape)


# In[ ]:


fig = px.scatter(train,x='LotFrontage',y='SalePrice',color='SalePrice',size='SalePrice')
fig.show()


# In[ ]:


train = train.drop(train[(train['LotFrontage']>300) & (train['SalePrice']<300000)].index)


# In[ ]:


fig = px.scatter(train,x='GrLivArea',y='SalePrice',size='SalePrice',color='SalePrice')
fig.show()


# In[ ]:


train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


# In[ ]:


fig  = px.scatter(train,x='LandSlope',y='SalePrice',size='SalePrice',color='SalePrice')
fig.show()


# In[ ]:


train = train.drop(train[(train['LandSlope']=='Gtl') & (train['SalePrice']>700000)].index)
train = train.drop(train[(train['LandSlope']=='Mod') & (train['SalePrice']>500000)].index)
train = train.drop(train[(train['LandSlope']=='Sev') & (train['SalePrice']>200000)].index)


# In[ ]:


train.shape


# In[ ]:


fig = px.scatter(train,x='Heating',y='SalePrice',size='SalePrice',color='SalePrice')
fig.show()


# <p style="color:magenta">No outliers.

# In[ ]:


fig = px.scatter(train,x='MSSubClass',y='SalePrice',size='SalePrice',color='SalePrice')
fig.show()


# In[ ]:


fig = px.scatter(train,x='MasVnrArea',y='SalePrice',size='SalePrice',color='SalePrice')
fig.show()


# In[ ]:


train = train.drop(train[(train['MasVnrArea']>1200)].index)


# **<h1 style="color:cyan">Tatrget Variable :**

# In[ ]:


plt.figure(figsize=(20,12))
plt.subplot(2,2,1)
sns.distplot(train['SalePrice'],color='green',bins=10)
plt.grid()
plt.title("Sale Price Values distribution")

sp = np.asarray(train['SalePrice'].values)
saleprice_transformed = stats.boxcox(sp)[0]

plt.subplot(2,2,2)
sns.distplot(saleprice_transformed,color='red',bins=10)
plt.grid()
plt.title("Box-Cox transformed Sale Price Values")

plt.show()


# In[ ]:


skewed_features = pd.DataFrame(train.skew().sort_values(ascending=False))
skewed_features = skewed_features.style.background_gradient(low=0,high=1,cmap='Purples',axis=0)
skewed_features


# In[ ]:


plt.figure(figsize=(25,20))
sns.heatmap(train.corr(),cmap='Oranges',fmt=".3f",annot=True)
plt.show()


# # **<h1 style="color:red">Data Modeling :**

# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


Y = train['SalePrice']
train = train.drop('SalePrice',axis=1)
data = pd.concat([train,test],axis=0)
data_ohe = pd.get_dummies(data)
train_ohe = data_ohe[:train.shape[0]]
test_ohe = data_ohe[train.shape[0]:]


# In[ ]:


print(train_ohe.shape)
print(test_ohe.shape)
print(Y.shape)


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(train_ohe,Y,test_size=0.2)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


X_train.isnull().sum()


# <h1 style="color:cyan">Scaling the data :

# In[ ]:


rbscaler = RobustScaler()
X_train = rbscaler.fit_transform(X_train)
X_test = rbscaler.fit_transform(X_test)
test_ohe = rbscaler.fit_transform(test_ohe)


# # **<h1 style="color:red">Models :**

# <h1 style="color:cyan">Linear Regression :

# In[ ]:


lr = LinearRegression()
lr.fit(X_train,Y_train)
train_pred = lr.predict(X_train)
pred = lr.predict(X_test)
print("Mean Squared Error on test data : ",mean_squared_error(Y_test,pred))
print("Mean Squared Error on train data : ",mean_squared_error(Y_train,train_pred))
rmse= np.sqrt(mean_squared_error(Y_test,pred))
rmse_train = np.sqrt(mean_squared_error(Y_train,train_pred))
print("Test rmse :",rmse)
print("Train rmse :",rmse_train)


# <h1 style="color:cyan">Lasso Regression :

# In[ ]:


# params = {
#     'alpha':[0.0001,0.001,0.01,0.1,0.2,0.3,0.311,0.4,1,10,100],
# }
# lasso = Lasso(normalize=True)

# clf = RandomizedSearchCV(lasso,params,n_jobs=-1,verbose=0,cv=10,scoring='neg_mean_squared_error')
# clf.fit(X_train,Y_train)

# print("Best parameters  :",clf.best_params_)


# In[ ]:


ls = Lasso(alpha=10,normalize=True)
ls.fit(X_train,Y_train)
train_pred = ls.predict(X_train)
test_pred = ls.predict(X_test)
print("Root Mean Square Error for train data is : ",np.sqrt(mean_squared_error(Y_train, train_pred)))
print("Root Mean Square Error test data is  : ",np.sqrt(mean_squared_error(Y_test, test_pred)))


# <h1 style="color:cyan">Ridge Regression :

# In[ ]:


# params = {
#     'alpha':[0.0001,0.001,0.01,0.1,0.2,0.3,0.311,0.4,1,10,100],
# }
# ridge = Ridge(normalize=True)

# clf = RandomizedSearchCV(ridge,params,n_jobs=-1,verbose=0,cv=10,scoring='neg_mean_squared_error')
# clf.fit(X_train,Y_train)

# print("Best parameters  :",clf.best_params_)


# In[ ]:


ridge = Ridge(alpha=0.1,normalize=True)
ridge.fit(X_train,Y_train)
train_pred = ridge.predict(X_train)
test_pred = ridge.predict(X_test)
print("Root Mean Square Error for train data is : ",np.sqrt(mean_squared_error(Y_train, train_pred)))
print("Root Mean Square Error test data is  : ",np.sqrt(mean_squared_error(Y_test, test_pred)))


# <h1 style="color:cyan">Elastic Net Regrsssion :

# In[ ]:


# params = {
#     'alpha':[0.0001,0.001,0.01,0.1,0.2,0.3,0.311,0.4,1,10,100],
#     'l1_ratio':[0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,10]
# }
# es = ElasticNet(normalize=True)

# clf = RandomizedSearchCV(es,params,n_jobs=-1,verbose=0,cv=10,scoring='neg_mean_squared_error')
# clf.fit(X_train,Y_train)

# print("Best parameters  :",clf.best_params_)


# In[ ]:


es = ElasticNet(alpha=0.001,l1_ratio=0.2)
es.fit(X_train,Y_train)
train_pred = es.predict(X_train)
test_pred = es.predict(X_test)
print("Root Mean Square Error for train data is : ",np.sqrt(mean_squared_error(Y_train, train_pred)))
print("Root Mean Square Error test data is  : ",np.sqrt(mean_squared_error(Y_test, test_pred)))


# <h1 style="color:cyan">XGBoost :

# In[ ]:


# xg_reg = xgb.XGBRegressor()
# xgparam_grid= {'learning_rate' : [0.01],'n_estimators':[2000, 3460, 4000],
#                                     'max_depth':[3], 'min_child_weight':[3,5],
#                                     'colsample_bytree':[0.5,0.7],
#                                     'reg_alpha':[0.0001,0.001,0.01,0.1,10,100],
#                                    'reg_lambda':[1,0.01,0.8,0.001,0.0001]}

# xg_grid=RandomizedSearchCV(xg_reg,xgparam_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# xg_grid.fit(X_train,Y_train)
# print(xg_grid.best_estimator_)
# print(xg_grid.best_score_)


# In[ ]:


xg = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
xg = xg.fit(X_train,Y_train)
train_pred = xg.predict(X_train)
pred = xg.predict(X_test)
print("Root Mean Square Error on train data is :",np.sqrt(mean_squared_error(Y_train, train_pred)))
print("Root Mean Square Error on test data is :",np.sqrt(mean_squared_error(Y_test, pred)))


# <h1 style="color:cyan">Light GBM :

# In[ ]:


# params = {
#     'learning_rate':[0.001,0.01,0.002,0.003,0.004,0.1,1,10],'n_estimators':[5,10,15,25,30,35,20,40,50,70,90,100,200,400,500,1000,1500,2000,5000],
#     'max_depth':[2,5,10,12,15,17,19,20,22,25,27,30,32,35,37,39,40,41,43,45,47,49,50,60,70,80,90,100,150,200],
#     'num_leaves' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
# }
# lg = lgb.LGBMRegressor()
# lg=RandomizedSearchCV(lg,params, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
# lg.fit(X_train,Y_train)
# print(lg.best_estimator_)
# print(lg.best_score_)


# In[ ]:


lg = lgb.LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1)
lg = lg.fit(X_train,Y_train)
train_pred = lg.predict(X_train)
pred = lg.predict(X_test)
print("Root Mean Square Error on train data is :",np.sqrt(mean_squared_error(Y_train, train_pred)))
print("Root Mean Square Error on test data is :",np.sqrt(mean_squared_error(Y_test, pred)))


# <h1 style="color:cyan">Gradient Boosting Decision Tree :

# In[ ]:


gbdt = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)
gbdt = gbdt.fit(X_train,Y_train)
train_pred = gbdt.predict(X_train)
pred = gbdt.predict(X_test)
print("Root Mean Square Error on train data is :",np.sqrt(mean_squared_error(Y_train, train_pred)))
print("Root Mean Square Error on test data is :",np.sqrt(mean_squared_error(Y_test, pred)))


# <h1 style="color:cyan">Voting Classifier :

# In[ ]:


vc = VotingRegressor([('LGBM',lg),('XGB',xg),('ElasticNet',es)])
vc = vc.fit(X_train,Y_train)
train_pred = vc.predict(X_train)
pred = vc.predict(X_test)
print("Root Mean Square Error on train data is :",np.sqrt(mean_squared_error(Y_train, train_pred)))
print("Root Mean Square Error on test data is :",np.sqrt(mean_squared_error(Y_test, pred)))


# <h1 style="color:red">Predictions :

# In[ ]:


test = test.reset_index(drop=True)
test['Id']


# In[ ]:


submit = pd.DataFrame(test['Id'],columns=['Id'])
predictions = vc.predict(test_ohe)
submit['SalePrice'] = predictions
len(submit)


# In[ ]:


submit.to_csv("submission.csv",index=False)
print("File Saved...")

