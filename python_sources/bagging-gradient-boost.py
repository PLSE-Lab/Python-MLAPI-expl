#!/usr/bin/env python
# coding: utf-8

# # Links
# 
# I got inspired by these kernels:
# 
# * https://www.kaggle.com/apapiu/regularized-linear-models
# * https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# * https://www.kaggle.com/learn/machine-learning
# 
# For stacking I use :
# https://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/
# 
# Bagging:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,RobustScaler,StandardScaler,Imputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from scipy.stats import skew
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import BaggingRegressor


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the currentd directory are saved as output.


# # Loading data
# 
# Loading data from train and test file. Test file provides only input data and I'll predict the prices via using a model.

# In[ ]:


def rmsle_cv(model, x, y):
    kf = KFold(10, shuffle=True, random_state=1).get_n_splits(x)
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kf, verbose=0))
    return (rmse)

def get_cat_cols(df):
    return  [col for col in df.columns if df[col].dtype == 'object']

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

y = np.log1p(train_data.SalePrice)
# test is meant for predictions and doesn't contain any price data. I need to provide it.
cand_train_predictors = train_data.drop(['Id', 'SalePrice'], axis=1)
cand_test_predictors = test_data.drop(['Id'], axis=1)

cat_cols = get_cat_cols(cand_train_predictors)

cand_train_predictors[cat_cols] = cand_train_predictors[cat_cols].fillna('NotAvailable')
cand_test_predictors[cat_cols] = cand_test_predictors[cat_cols].fillna('NotAvailable')

encoders = {}

for col in cat_cols:
    encoders[col] = LabelEncoder()
    val = cand_train_predictors[col].tolist()
    val.extend(cand_test_predictors[col].tolist())
    encoders[col].fit(val)
    cand_train_predictors[col] = encoders[col].transform(cand_train_predictors[col])
    cand_test_predictors[col] = encoders[col].transform(cand_test_predictors[col])

    
corr_matrix = cand_train_predictors.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
cols_to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
print('Highly correlated features(will be droped):',cols_to_drop)

cand_train_predictors = cand_train_predictors.drop(cols_to_drop, axis=1)
cand_test_predictors = cand_test_predictors.drop(cols_to_drop, axis=1)

print(cand_train_predictors.shape)
print(cand_test_predictors.shape)

cand_train_predictors.fillna(cand_train_predictors.mean(), inplace=True)
cand_test_predictors.fillna(cand_test_predictors.mean(), inplace=True)

skewed_feats = cand_train_predictors.apply(lambda x: skew(x))  # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

print('Skewed features:', skewed_feats)

cand_train_predictors[skewed_feats] = np.log1p(cand_train_predictors[skewed_feats])
cand_test_predictors[skewed_feats] = np.log1p(cand_test_predictors[skewed_feats])

train_set, test_set = cand_train_predictors.align(cand_test_predictors,join='left', axis=1)


# # Model
# 

# In[ ]:


gdr = BaggingRegressor(base_estimator=GradientBoostingRegressor(n_estimators=1000))
gdr_model = gdr
print(gdr_model)
gdr_model.fit(train_set, y)

print('score gradient boost:', gdr_model.score(train_set, y))

train_pred = gdr_model.predict(train_set)

print('rmse from log: ', np.sqrt(mean_squared_error(y, train_pred)))
print('mse from log: ', mean_squared_error(y, train_pred))
print('rmsle: ', np.sqrt(mean_squared_log_error(y, train_pred)))
print('rmse: ', np.sqrt(mean_squared_error(train_data.SalePrice, np.expm1(train_pred))))
print('mse: ', mean_squared_error(train_data.SalePrice, np.expm1(train_pred)))
print('mae: ', mean_absolute_error(train_data.SalePrice, np.expm1(train_pred)))


# # Predicting and submitting
# 
# Now it's time to predict from test.

# In[ ]:


test_pred = gdr_model.predict(test_set)
predicted_prices = np.expm1(test_pred)
print(predicted_prices[:5])

# print(len(predicted_prices))
# print(len(test_data.Id))

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
my_submission.Id = my_submission.Id.astype(int)
# print(my_submission.Id)
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

