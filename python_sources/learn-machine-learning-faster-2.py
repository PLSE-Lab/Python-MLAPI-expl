#!/usr/bin/env python
# coding: utf-8

# # This is the second notebook of the Machine Learning series faster
# Enjoy learning

# first import the main Library

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In this notebook we will talk about four very important algorithms as well as some other important algorithms 
# 
# Light GBM
# 
# XGBoost
#                  
#  Catboost
#                  
#  Stochastic Gradient Descent
#               
#   Lasso

# ![images%20%282%29.jpg](attachment:images%20%282%29.jpg)

# # Light GBM

# * **LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:**
# 
# 1. Faster training speed and higher efficiency.
# 2. Lower memory usage.
# 3. Better accuracy.
# 4. Support of parallel and GPU learning.
# 5. Capable of handling large-scale data.
# 
# ![build-an-efficient-machine-learning-model-with-lightgbm-29-638.jpg](attachment:build-an-efficient-machine-learning-model-with-lightgbm-29-638.jpg)
# 

# In[ ]:


import lightgbm as lgbm
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import r2_score


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
data = pd.concat([train, test], sort=False)
data = data.reset_index(drop=True)
data.head()


# > **Preprocessing**

# In[ ]:


nans=pd.isnull(data).sum()

data['MSZoning']  = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
data['Utilities'] = data['Utilities'].fillna(data['Utilities'].mode()[0])
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

data["BsmtFinSF1"]  = data["BsmtFinSF1"].fillna(0)
data["BsmtFinSF2"]  = data["BsmtFinSF2"].fillna(0)
data["BsmtUnfSF"]   = data["BsmtUnfSF"].fillna(0)
data["TotalBsmtSF"] = data["TotalBsmtSF"].fillna(0)
data["BsmtFullBath"] = data["BsmtFullBath"].fillna(0)
data["BsmtHalfBath"] = data["BsmtHalfBath"].fillna(0)
data["BsmtQual"] = data["BsmtQual"].fillna("None")
data["BsmtCond"] = data["BsmtCond"].fillna("None")
data["BsmtExposure"] = data["BsmtExposure"].fillna("None")
data["BsmtFinType1"] = data["BsmtFinType1"].fillna("None")
data["BsmtFinType2"] = data["BsmtFinType2"].fillna("None")

data['KitchenQual']  = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data["Functional"]   = data["Functional"].fillna("Typ")
data["FireplaceQu"]  = data["FireplaceQu"].fillna("None")

data["GarageType"]   = data["GarageType"].fillna("None")
data["GarageYrBlt"]  = data["GarageYrBlt"].fillna(0)
data["GarageFinish"] = data["GarageFinish"].fillna("None")
data["GarageCars"] = data["GarageCars"].fillna(0)
data["GarageArea"] = data["GarageArea"].fillna(0)
data["GarageQual"] = data["GarageQual"].fillna("None")
data["GarageCond"] = data["GarageCond"].fillna("None")

data["PoolQC"] = data["PoolQC"].fillna("None")
data["Fence"]  = data["Fence"].fillna("None")
data["MiscFeature"] = data["MiscFeature"].fillna("None")
data['SaleType']    = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['LotFrontage'].interpolate(method='linear',inplace=True)
data["Electrical"]  = data.groupby("YearBuilt")['Electrical'].transform(lambda x: x.fillna(x.mode()[0]))
data["Alley"] = data["Alley"].fillna("None")

data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
nans=pd.isnull(data).sum()
nans[nans>0]


# In[ ]:


_list = []
for col in data.columns:
    if type(data[col][0]) == type('str'): 
        _list.append(col)

le = preprocessing.LabelEncoder()
for li in _list:
    le.fit(list(set(data[li])))
    data[li] = le.transform(data[li])

train, test = data[:len(train)], data[len(train):]

X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']

test = test.drop(columns=['SalePrice', 'Id'])


# **Model and Accuracy**

# In[ ]:


kfold = KFold(n_splits=5, random_state = 2020, shuffle = True)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X, y)
r2_score(model_lgb.predict(X), y)


# > > # **XGBoost**

# **XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In prediction problems involving unstructured data (images, text, etc.) artificial neural networks tend to outperform all other algorithms or frameworks.It is a perfect combination of software and hardware optimization techniques to yield superior results using less computing resources in the shortest amount of time.**
# ![1_1kjLMDQMufaQoS-nNJfg1Q.png](attachment:1_1kjLMDQMufaQoS-nNJfg1Q.png)

# **Library and Data**

# In[ ]:


import xgboost as xgb
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_xgb.fit(X, y)
r2_score(model_xgb.predict(X), y)


# # Catboost

# **Catboost is a type of gradient boosting algorithms which can  automatically deal with categorical variables without showing the type conversion error, which helps you to focus on tuning your model better rather than sorting out trivial errors.Make sure you handle missing data well before you proceed with the implementation.
# **
# ![boosting-approach-to-solving-machine-learning-problems-15-638.jpg](attachment:boosting-approach-to-solving-machine-learning-problems-15-638.jpg)

# **Library and Data**

# In[ ]:


from catboost import CatBoostRegressor
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.05,
                             depth=10,
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
cb_model.fit(X, y)
r2_score(cb_model.predict(X), y)


# # Stochastic Gradient Descent

# **Stochastic means random , so in Stochastic Gradient Descent dataset sample is choosedn random instead of the whole dataset.hough, using the whole dataset is really useful for getting to the minima in a less noisy or less random manner, but the problem arises when our datasets get really huge and for that SGD come in action**
# ![an-overview-of-gradient-descent-optimization-algorithms-13-638.jpg](attachment:an-overview-of-gradient-descent-optimization-algorithms-13-638.jpg)
# 
# ![image.png](attachment:image.png)

# **Library and Data**

# In[ ]:


from sklearn.linear_model import SGDRegressor
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


SGD = SGDRegressor(max_iter = 100)
SGD.fit(X, y)
r2_score(SGD.predict(X), y)


# # Lasso
# **In statistics and machine learning, lasso (least absolute shrinkage and selection operator; also Lasso or LASSO) is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces. Though originally defined for least squares, lasso regularization is easily extended to a wide variety of statistical models including generalized linear models, generalized estimating equations, proportional hazards models, and M-estimators, in a straightforward fashion**
# 

# **Library and Data**

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso.fit(X, y)
r2_score(lasso.predict(X), y)


# # Kernel Ridge Regression

# **KRR combine Ridge regression and classification with the kernel trick.It is similar to Support vector Regression but relatively very fast.This is suitable for smaller dataset (less than 100 samples)**

# **Library and Data**

# In[ ]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
KRR.fit(X, y)
r2_score(KRR.predict(X), y)


# # BayesianRidge

# ** Bayesian regression, is a regression model defined in probabilistic terms, with explicit priors on the parameters. The choice of priors can have the regularizing effect.Bayesian approach is a general way of defining and estimating statistical models that can be applied to different models.**

# **Library and Data**

# In[ ]:


from sklearn.linear_model  import BayesianRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


BR = BayesianRidge()
BR.fit(X, y)
r2_score(BR.predict(X), y)


# # Elastic Net Regression 
# 

# **Elastic net is a hybrid of ridge regression and lasso regularization.It combines feature elimination from Lasso and feature coefficient reduction from the Ridge model to improve your model's predictions.**
# 

# **Library and Data**

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
#Data is used the same as LGB
X = train.drop(columns=['SalePrice', 'Id']) 
y = train['SalePrice']
X.head()


# **Model and Accuracy**

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
ENet.fit(X, y)
r2_score(ENet.predict(X), y)

