#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import VotingClassifier, BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn import tree
import xgboost as xgb
import lightgbm as lgb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


houseDataTrain = pd.read_csv("../input/train.csv")
houseDataTest = pd.read_csv("../input/test.csv")
houseDataSampleSubmission = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


houseDataTrain.head(10)


# In[ ]:


print(houseDataTrain.shape)


# In[ ]:


houseDataTest.head(5)


# In[ ]:


print(houseDataTest.shape)


# In[ ]:


houseDataSampleSubmission.head(5)


# In[ ]:


houseDataTrain.describe()


# In[ ]:


houseDataTrain.isnull().sum().sum()


# In[ ]:


nullColumns = houseDataTrain.columns[houseDataTrain.isnull().any()]


# In[ ]:


print(nullColumns)


# In[ ]:


for item in nullColumns:
    if houseDataTrain[item].dtype == 'float64':
        houseDataTrain[item].fillna((houseDataTrain[item].mean()), inplace = True)
    elif houseDataTrain[item].dtype == 'O':
        houseDataTrain[item].fillna(houseDataTrain[item].value_counts().index[0], inplace = True)


# In[ ]:


houseDataTrain.isnull().sum().sum()


# In[ ]:


houseDataTrain[nullColumns].head(10)


# In[ ]:


objectColumns = houseDataTrain.select_dtypes(['object']).columns


# In[ ]:


objectColumns


# In[ ]:


for item in objectColumns:
    houseDataTrain[item] = houseDataTrain[item].astype('category')


# In[ ]:


categoryColumns = houseDataTrain.select_dtypes(['category']).columns


# In[ ]:


houseDataTrain[categoryColumns] = houseDataTrain[categoryColumns].apply(lambda x: x.cat.codes)


# In[ ]:


houseDataTrain.head(10)


# In[ ]:


y = houseDataTrain['SalePrice']


# In[ ]:


x = houseDataTrain.drop(['Id', 'SalePrice'], axis = 1)


# In[ ]:


print(x.shape)


# In[ ]:


print(y.shape)


# In[ ]:


x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size = 0.25, random_state = 42)


# In[ ]:


print(x_train.shape)
print(x_cv.shape)


# In[ ]:


linearRegression = LinearRegression()


# In[ ]:


linearRegression.fit(x_train, y_train)


# In[ ]:


linearRegression.score(x_train, y_train)


# In[ ]:


prediction_linearRegression = linearRegression.predict(x_cv)


# In[ ]:


mean_squared_error(y_cv, prediction_linearRegression)


# In[ ]:


print(linearRegression.score(x_cv, y_cv))
print(r2_score(y_cv, prediction_linearRegression))


# In[ ]:


ridgeRegression = Ridge()


# In[ ]:


ridgeRegression.fit(x_train, y_train)


# In[ ]:


ridgeRegression.score(x_train, y_train)


# In[ ]:


prediction_ridgeRegression = ridgeRegression.predict(x_cv)


# In[ ]:


mean_squared_error(y_cv, prediction_ridgeRegression)


# In[ ]:


r2_score(y_cv, prediction_ridgeRegression)


# In[ ]:


lassoRegression = Lasso(alpha = 1, max_iter = 5000)


# In[ ]:


lassoRegression.fit(x_train, y_train)


# In[ ]:


lassoRegression.score(x_train, y_train)


# In[ ]:


prediction_lassoRegression = lassoRegression.predict(x_cv)


# In[ ]:


mean_squared_error(y_cv, prediction_lassoRegression)


# In[ ]:


r2_score(y_cv, prediction_lassoRegression)


# In[ ]:


elasticNet = ElasticNet(alpha = 1, l1_ratio = 0.9, max_iter = 5000, normalize = False)


# In[ ]:


elasticNet.fit(x_train, y_train)


# In[ ]:


elasticNet.score(x_train, y_train)


# In[ ]:


prediction_elasticNet = elasticNet.predict(x_cv)


# In[ ]:


mean_squared_error(y_cv, prediction_elasticNet)


# In[ ]:


r2_score(y_cv, prediction_elasticNet)


# In[ ]:


votingClassifier = VotingClassifier(estimators = [('Linear Regression', linearRegression), ('Ridge Regression', ridgeRegression), ('Lasso Regression', lassoRegression), ('Elastic Net Regression', elasticNet)], voting = 'hard')


# In[ ]:


baggingRegressor = BaggingRegressor(tree.DecisionTreeRegressor(random_state = 1))


# In[ ]:


baggingRegressor.fit(x_train, y_train)


# In[ ]:


baggingRegressor.score(x_train, y_train)


# In[ ]:


prediction_baggingRegressor = baggingRegressor.predict(x_cv)


# In[ ]:


mean_squared_error(y_cv, prediction_baggingRegressor)


# In[ ]:


r2_score(y_cv, prediction_baggingRegressor)


# In[ ]:


randomForestRegressor = RandomForestRegressor(n_estimators = 30)


# In[ ]:


randomForestRegressor.fit(x_train, y_train)


# In[ ]:


randomForestRegressor.score(x_train, y_train)


# In[ ]:


prediction_randomForest = randomForestRegressor.predict(x_cv)


# In[ ]:


mean_squared_error(y_cv, prediction_randomForest)


# In[ ]:


r2_score(y_cv, prediction_randomForest)


# In[ ]:


adaBoostRegressor = AdaBoostRegressor(n_estimators = 60)


# In[ ]:


adaBoostRegressor.fit(x_train, y_train)


# In[ ]:


adaBoostRegressor.score(x_train, y_train)


# In[ ]:


prediction_adaBoost = adaBoostRegressor.predict(x_cv)


# In[ ]:


mean_squared_error(y_cv, prediction_adaBoost)


# In[ ]:


r2_score(y_cv, prediction_adaBoost)


# In[ ]:


gradientBoostingRegressor = GradientBoostingRegressor(max_depth = 4)


# In[ ]:


gradientBoostingRegressor.fit(x_train, y_train)


# In[ ]:


gradientBoostingRegressor.score(x_train, y_train)


# In[ ]:


prediction_gradientBoost = gradientBoostingRegressor.predict(x_cv)


# In[ ]:


mean_squared_error(y_cv, prediction_gradientBoost)


# In[ ]:


r2_score(y_cv, prediction_gradientBoost)


# In[ ]:


xgBoost = xgb.XGBRegressor(max_depth = 4, learning_rate = 0.1, n_estimators = 500)


# In[ ]:


xgBoost.fit(x_train, y_train)


# In[ ]:


xgBoost.score(x_train, y_train)


# In[ ]:


prediction_xgBoost = xgBoost.predict(x_cv)


# In[ ]:


mean_squared_error(y_cv, prediction_xgBoost)


# In[ ]:


r2_score(y_cv, prediction_xgBoost)


# In[ ]:


params = {'learning_rate': 0.1}
train_data = lgb.Dataset(x_train, label = y_train)
lgbRegressor = lgb.train(params, train_data, 100)


# In[ ]:


prediction_lgbRegressor = lgbRegressor.predict(x_cv)


# In[ ]:


mean_squared_error(y_cv, prediction_lgbRegressor)


# In[ ]:


r2_score(y_cv, prediction_lgbRegressor)

