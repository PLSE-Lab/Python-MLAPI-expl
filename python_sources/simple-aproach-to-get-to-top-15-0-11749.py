#!/usr/bin/env python
# coding: utf-8

# # Simple aproach to get to TOP 15% of the LB (0.11749) #

# ## Alejandro Guerra ##
# ** October, 2017 **

# This kernel is my simple approach to get to TOP 15% of the LB. *House Prices: Advanced Regression Techniques* is my first Kaggle competition so I will appreciate so much your suggestions.
# 
# To develop this kernel I have read several useful kernels from other users. For example:
# * [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard) by [Sergine](https://www.kaggle.com/serigne)
# * [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) by [Pedro Marcelino](https://www.kaggle.com/pmarcelino)
# * [Regularized Linear Models](https://www.kaggle.com/apapiu/regularized-linear-models) by [Alexandru Papiu](https://www.kaggle.com/apapiu)
# 
# Also, it's very useful read the [documentation](http://ww2.amstat.org/publications/jse/v19n3/Decock/DataDocumentation.txt) of Ames Housing Data. Tasks like remove outliers and handle missing values are more easy and accurate if you read the documentation.
# 
# In this notebook, I use 4 diferents models: Lasso Regression, Elastic Net Regrssion, Kernel Ridge Regression and XGBoost. First, I use GridSearch to tune model parameters and then I make the predictions averaging the predictions of the 4 models.
# 
# The notebook is structured as follows:
# * **Data visualization and feature engeeniering:** 
# 
#      * Remove outliers
#      * Missing values imputation
#      * Create aditional features
#      * Log transformation of the target variable
#      * Transform skeewed features
#      * Encode categorical features
# * **Parameter tunning**
# * **Make predicctions**
# 
# Thanks for reading!

# ## Basic imports and get the data

# In[ ]:


import time
time1 = time.time()


# In[ ]:


# Basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Import data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ## Data visualization and feature engeeniering

# First, get the numbers of examples and features and its types.

# In[ ]:


# Get shape
print("Shape of train dataset: ", train.shape)
print("Shape of test dataset:  ", test.shape)
# Get numbers of training examples...
print("Training examples:      ", train.shape[0])
print("Test examples:          ", test.shape[0])
print("Features:               ", test.shape[1])
# Types of the data
print("Numerical features:     ", test.dtypes.value_counts()[1]+test.dtypes.value_counts()[2])
print("Categorical features:   ", test.dtypes.value_counts()[0])


# ### Remove Outliers ###
# 
# Documentation for the Ames dataset says:
# *There are 5 observations that an instructor may wish to remove from the data set before giving it to students (a plot of SALE PRICE versus GR LIV AREA will indicate them quickly)....*
# 
# Let's get the graph:

# In[ ]:


# Plot SalePrice vs GrLivArea
plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()


# As we can see at the right, the data has two clear outliers. Let's remove them and check the results.

# In[ ]:


# Delete Outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
# Plot SalePrice vs GrLivArea
plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()


#  To continue the analysis concatenate training and test examples.

# In[ ]:


# Concatenate train and test datasets
data = pd.concat((train, test),ignore_index=True)
data = data.drop("SalePrice",1)


# ### Missing Values Imputation###
# 
# Check the number of missing values:

# In[ ]:


s1 = data.isnull().sum()[data.isnull().sum() != 0]
s2 = s1/data.shape[0]
pd.concat({'Missing Values': s1,
           'Missing Ratio': s2},axis=1).sort_values(by = 'Missing Ratio', ascending = False)


# To impute missing values is crucial read the Ames Housing Data documentation. Most missing values correspond to 0 or None.

# In[ ]:


# Missing Values imputation
# Replace NaN with 0
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'GarageCars',
           'GarageArea', 'MasVnrArea', 'BsmtHalfBath'):
    data[col] = data[col].fillna(0)

# Replace NaN with median
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())

# Replace with None
for col in ('PoolQC', 'Fence', 'FireplaceQu','GarageType','GarageFinish', 'GarageQual',
            'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'MSSubClass', 'MiscFeature'):
    data[col] = data[col].fillna('None')
    
# Replace with mode
for col in ('Exterior1st', 'Exterior2nd', 'KitchenQual', 'Electrical', 'SaleType',
            'MSZoning', 'GarageYrBlt'):
    data[col] =  data[col].fillna(data[col].mode()[0])
    
# Replace with especific value
data["Functional"] = data["Functional"].fillna("Typ")

# Delete some features
data = data.drop(['Id', 'Utilities'],1)


# ### Create TotalSF feature ###
# Create a feature that computes the total area of the house.

# In[ ]:


# Create total square feets feature
TotalSF = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']
data.insert(loc=0,column="TotalSF", value=TotalSF)


# ### Log transformation of the target variable###

# Get the histogram plot of the target variable

# In[ ]:


Y = train.SalePrice
sns.distplot(Y)


# The target variable is right skeewed. Let's transform it! We use log(1+x).

# In[ ]:


# Log transform target variable
y_log = np.log1p(Y)


# Check the result

# In[ ]:


sns.distplot(y_log)


# ### Transform skeewed features ###
# Also use log(1+x)

# In[ ]:


# Transform skewed features
skew = abs(data.skew()).sort_values(ascending = False)
skew_features = skew[skew > 1].index

data[skew_features] = np.log1p(data[skew_features])


# ### Encode categorical features ###
# I use dummy encoding. It works better than label encoding.

# In[ ]:


data = pd.get_dummies(data)


# In[ ]:


# Split in train an test datasets
X_train = data[:train.shape[0]]
X_test = data[train.shape[0]:]


# ## Model parameter tunning ## 

# In this section I use GridSearch to get the best params for each model. First, import the models:

# In[ ]:


# Imports
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV


# In[ ]:


# Parameter tunning Lasso
model = Lasso()
grid = GridSearchCV(estimator=model, param_grid={'alpha': [100,10,1,0.1,0.01,0.001]})
grid.fit(X_train, y_log)

print('Lasso Regression parameters:')
print('Alpha: ',grid.best_estimator_.alpha)


# In[ ]:


# Parameter tunning ElasticNet
model = ElasticNet()
grid = GridSearchCV(estimator=model, param_grid={'alpha': [100,10,1,0.1,0.01,0.001],
                                                 'l1_ratio': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]})
grid.fit(X_train, y_log)

print('Elastic Net parameters:')
print('Alpha: ',grid.best_estimator_.alpha)
print('l1_ratio: ',grid.best_estimator_.l1_ratio)


# In[ ]:


# Parameter tunning Kernel Ridge Regression
model = KernelRidge(kernel='polynomial')
grid = GridSearchCV(estimator=model, param_grid={'alpha': [1000,10,1],
                                                 'coef0': [1000,10,1],
                                                 'degree': [2,3]})
grid.fit(X_train, y_log)

print('Kernel Ridge Regression parameters:')
print('alpha: ',grid.best_estimator_.alpha)
print('coef0: ',grid.best_estimator_.coef0)
print('degree: ',grid.best_estimator_.degree)


# In[ ]:


# Parameter tunning XGB
model = xgb.XGBRegressor()
grid = GridSearchCV(estimator=model, param_grid={'n_estimators': [300,400,500,600,700],
                                                 'max_depth': [2,3],
                                                 'learning_rate': [0.05,0.1]})
grid.fit(X_train, y_log)

print('XGB Regressor parameters:')
print('alpha: ',grid.best_estimator_.n_estimators)
print('max_depth: ',grid.best_estimator_.max_depth)
print('degree: ',grid.best_estimator_.learning_rate)


# ## Make predictions! ##

# ### Cross validate models ###

# In[ ]:


# Cross - Validation score (Credits: Sergine)
def rmsle_cv(model):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_log, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[ ]:


# Lasso
model_lasso = Lasso(alpha=0.001)

score = rmsle_cv(model_lasso)
print("Lasso Regression Score: {:.4f}".format(score.mean()))


# In[ ]:


# ElasticNet
model_enet = ElasticNet(alpha=0.001,l1_ratio=0.5)

score = rmsle_cv(model_enet)
print("ElasticNet Score: {:.4f}".format(score.mean()))


# In[ ]:


# Kernel Ridge Regressor
model_krr = KernelRidge(alpha=1000, coef0=1000, degree=2, kernel='polynomial')

score = rmsle_cv(model_krr)
print("Kernel Ridge Regressor Score: {:.4f}".format(score.mean()))


# In[ ]:


# XGBoost Regressor
model_xgb = xgb.XGBRegressor(n_estimators=700, max_depth=2, learning_rate=0.05)

score = rmsle_cv(model_xgb)
print("XGBoost Score: {:.4f}".format(score.mean()))


# ### Average models ###

# In[ ]:


model_krr.fit(X_train,y_log)
p_krr = np.expm1(model_krr.predict(X_test))

model_enet.fit(X_train,y_log)
p_enet = np.expm1(model_enet.predict(X_test))

model_lasso.fit(X_train,y_log)
p_lasso = np.expm1(model_lasso.predict(X_test))

model_xgb.fit(X_train,y_log)
p_xgb = np.expm1(model_xgb.predict(X_test))


# In[ ]:


# Ensemble 4 Models (Average)
p_average = (p_lasso + p_xgb + p_krr + p_enet)/4

out = pd.Series(p_average,index=test.loc[:,'Id'])
out.name = 'SalePrice'
out.to_csv('average.csv', header=True, index_label='Id')


# In[ ]:


time2 = time.time()
print('Time: ', time2-time1)

