#!/usr/bin/env python
# coding: utf-8

# In this kernel, we will do:
# * Basic data exploration
# * Trying to create new derived features
# * Handle missing values and use one hot encoding for the categorical variables.
# * Try out various models-
#     1. Linear Regression
#     2. Lasso
#     3. Decision Tree Regressor
#     4. SVR
#     5. Gradient Boosting Regressor
# * Use ensembling for the final output.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)


import matplotlib.pyplot as plt
import seaborn as sns

from ml_metrics import rmse

#importing models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
#from sklearn.tree import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# importing all the data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test2 = pd.read_csv("../input/test.csv")
houses = pd.concat([train,test],sort=False)
print(train.shape)
print(test.shape)


# Let's look at categorical and numerical features separately

# In[ ]:


houses.select_dtypes(include='object').head()


# In[ ]:


houses.select_dtypes(exclude='object').head()


# Looks like MSSubclass is actually a categorical variable and should be treated accordingly
# 
# Let's look at null values first and understand deapth of data
# Categorical first

# In[ ]:


train.select_dtypes(include='object').isnull().sum()


# In[ ]:


train.select_dtypes(exclude='object').isnull().sum()


# Looks like we have most of the data missing for following categorical variables. Probably it is better to get rid of these columns as they might not have predictive power. Let's remove.
# Alley, PoolQC, Fense, MiscFeature

# In[ ]:


train.drop(['Alley', 'PoolQC','Fence', 'MiscFeature'], axis=1, inplace=True)
test.drop(['Alley', 'PoolQC','Fence', 'MiscFeature'], axis=1, inplace=True)


# BsmtQual, BsmtCond, BsmtExposure,BsmtFinType1, BsmtFinType2, FireplaceQu, GarageType, GarageFinish, Garagequal, GarageCond, PavedDrive are other categorical variables missing some values. Let's replace. 

# In[ ]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure','BsmtFinType1', 
            'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
            'GarageQual', 'GarageCond', 'PavedDrive', 'Electrical','MasVnrType'):
    train[col]=train[col].fillna('Unknown')
    test[col]=test[col].fillna('Unknown')


# Making sure all categorical missing values have been filled

# In[ ]:


print(train.GarageType.isnull().sum())
print(test.GarageType.isnull().sum())


# Now lets handle missing values of numerical features

# In[ ]:


train['LotFrontage']=train['LotFrontage'].fillna(train.LotFrontage.mean())
test['LotFrontage']= test['LotFrontage'].fillna(test.LotFrontage.mean())
train['MasVnrArea']= train['MasVnrArea'].fillna(train.MasVnrArea.mean())
test['MasVnrArea']= test['MasVnrArea'].fillna(test.MasVnrArea.mean())
train['GarageYrBlt']= train['GarageYrBlt'].fillna(0)
test['GarageYrBlt']= test['GarageYrBlt'].fillna(0)

test['BsmtFinSF1']= test['BsmtFinSF1'].fillna(test.BsmtFinSF1.mean())
test['BsmtFinSF2']= test['BsmtFinSF2'].fillna(test.BsmtFinSF2.mean())
test['BsmtUnfSF']= test['BsmtUnfSF'].fillna(test.BsmtUnfSF.mean())
test['BsmtFullBath']= test['BsmtFullBath'].fillna(test.BsmtFullBath.mean())
test['BsmtHalfBath']= test['BsmtHalfBath'].fillna(test.BsmtHalfBath.mean())
test['GarageCars']= test['GarageCars'].fillna(test.GarageCars.mean())
test['TotalBsmtSF']= test['TotalBsmtSF'].fillna(test.TotalBsmtSF.mean())
test['GarageArea'] = test['GarageArea'].fillna(test.GarageArea.mean())


# Let's make sure nothing null is left

# In[ ]:



test.columns[test.isnull().any()].tolist()


# Getting to understand correlations with price for for various variables by plotting correlation heatmap

# In[ ]:


plt.figure(figsize=[80,40])
sns.heatmap(train.corr(),annot=True)


# plotting heatmap of most correlated features which look useful.
# The value of 0.3 was chosen after trying out numerous values.

# In[ ]:


corr = train.corr()
most_corr_features = corr.index[abs(corr["SalePrice"])>0.3]
plt.figure(figsize=(10,10))
sns.heatmap(train[most_corr_features].corr(), annot=True, cmap="RdYlGn")


# In[ ]:


#removing outliers recomended by author
train = train[train['GrLivArea']<4000]


# In[ ]:


train.describe()


# In[ ]:


print(train.select_dtypes(exclude='object').columns)

#These features are highly correlated with other features and themselves have lower correlation with SalePrice
#'GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'


# It's time to build the model

# In[ ]:


features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']
x_train, x_test, y_train, y_test = train_test_split(train[features],train['SalePrice'], train_size = 0.8, 
                                                    test_size = 0.2, random_state=3)
lm = LinearRegression()
lm.fit(x_train, y_train)
print(lm.score(x_train, y_train))
print(lm.score(x_test, y_test))


# basic linear regression. The r2 score is not very good

# In[ ]:


most_corr_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea',
       'FullBath', 'Fireplaces', 'GarageCars']
steps = [
   ('scalar', StandardScaler()),
   ('poly', PolynomialFeatures(degree=3)),
   ('model', Lasso(alpha=1000, fit_intercept=True))
]
pipeline = Pipeline(steps)

x_train2, x_test2, y_train2, y_test2 = train_test_split(train[most_corr_features], train["SalePrice"], train_size=0.8, test_size=0.2, random_state=3)

pipeline.fit(x_train2, y_train2)
print(pipeline.score(x_train2, y_train2))
print(pipeline.score(x_test2, y_test2))


# trying out lasso model. Lasso model is better than plain linear regression.

# In[ ]:


len_train=train.shape[0]
houses=pd.concat([train,test], sort=False)

for col in ('MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType'):
    houses[col]=houses[col].fillna('Unknown')



# filling categorical features' missing values with Unknown as it may still have some data hidden.

# In[ ]:


houses["MSSubClass"] = houses["MSSubClass"].apply(str)
#houses["OverallCond"] = houses["OverallCond"].apply(str)

houses["MoSold"] = houses["MoSold"].apply(str)


# converting mosold and subclass to categorical as it doesn't make sense to keep them linear.

# In[ ]:


#add any new features you make over here.
#houses["yearDiff"] = houses["YearRemodAdd"] - houses["YearBuilt"]
#houses["IsRemod"] = houses.apply(lambda row: 0 if row['yearDiff'] == 0 else 1,axis=1)
#houses["HasGarage"] = houses.apply(lambda row: 0 if row['GarageArea'] == 0 else 1,axis=1)
houses["YrSold"] = houses["YrSold"].apply(str)


# tried many derived features but they didn't add anything. Converting year sold to string as well.

# In[ ]:


#houses.drop("yearDiff", axis=1, inplace=True)
houses.drop("GarageArea", axis=1, inplace=True)

houses=pd.get_dummies(houses)
train=houses[:len_train]
test=houses[len_train:]
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
train['SalePrice']=np.log(train['SalePrice'])*100
x=train.drop('SalePrice', axis=1)
y=train['SalePrice']
test=test.drop('SalePrice', axis=1)


# garage area is making the model worse so we will remove it. Also for all the categorical features we have done one hot encoding.

# In[ ]:



steps = [
   
   ('poly', PolynomialFeatures(degree=1)),
   ('model', Lasso(alpha=0.15, fit_intercept=True))
]
pipeline = Pipeline(steps)

x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=3)

pipeline.fit(x_train2, y_train2)
#print(pipeline.score(x_train2, y_train2))
#print(pipeline.score(x_test2, y_test2))
print(rmse(actual=y_test2/100, predicted=pipeline.predict(x_test2)/100))


# fitting our final lasso model after tuning our hyper parameters.
# Also, we are now going to predict the log of Sale Price as that is what is being used in the evaluation.

# In[ ]:


#regressor = DecisionTreeRegressor(random_state = 0)
#regressor.fit(x_train2, y_train2)
#print(rmse(actual=y_test2/100, predicted=regressor.predict(x_test2)/100))
#print(rmse(actual=y_test2/100, predicted=(pipeline.predict(x_test2) + regressor.predict(x_test2))/200))
#train["predictions"] = (pipeline.predict(x)/100)
#train["error"] = train.apply(lambda row: row["predictions"] - (row["SalePrice"]/100), axis=1)


# Tried out decision tree regressor but it did much worse.

# In[ ]:


#model = SVR(epsilon=0.01, kernel='poly')
#print(model)
#model.fit(x_train2, y_train2)
#print(rmse(actual=y_test2/100, predicted=model.predict(x_test2)/100))


# tried svr also but it did much worse. Also using the poly kernel takes too long to run.

# In[ ]:


predictions1 = pipeline.predict(test)


# making the first set of predictions using lasso.

# In[ ]:


params = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.1,
    'criterion': 'mae',
    'min_impurity_decrease': 0.005,
    'loss': 'huber',
    'alpha': 0.99
}

gradient_boosting_regressor = GradientBoostingRegressor(**params)
gradient_boosting_regressor.fit(x_train2, y_train2)
print(rmse(actual=y_test2/100, predicted=gradient_boosting_regressor.predict(x_test2)/100))


# Tried the gradient boosting model and it does very well after training the hyper-parameters. It does almost as good as the lasso model, so now we will try ensembling the two models.

# In[ ]:


predictions2 = gradient_boosting_regressor.predict(test)
predictions = np.exp((predictions1 + predictions2)/200)
my_sub = pd.DataFrame({'Id':test2.Id, 'SalePrice':predictions})
my_sub.to_csv('submission.csv', index = False)


# Taking average of gradient boosting and lasso predictions to give the final predictions.
