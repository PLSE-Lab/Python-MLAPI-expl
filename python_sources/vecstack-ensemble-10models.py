#!/usr/bin/env python
# coding: utf-8

#         This program uses vecstack ensemble for stacking different models.  https://github.com/vecxoz/vecstack
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor

from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.linear_model import RidgeCV,LassoCV,LassoLarsCV

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from vecstack import stacking
import lightgbm as lgb

import pandas as pd
import numpy as np
import os


# In[ ]:


# give the path to where the training and test set are present
train = pd.read_csv('../input/santander-value-prediction-challenge/train.csv')

test = pd.read_csv('../input/santander-value-prediction-challenge/test.csv')

target_val = train['target']
train['log_target'] = np.log1p(target_val)

test_ID = test['ID'].values


# In[ ]:


X_train, X_test = train_test_split(train, test_size=0.2, random_state=5)


# In[ ]:


y_train = X_train['log_target']
y_test = X_test['log_target']

train.drop("ID", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)
train.drop("target", axis = 1, inplace = True)
train.drop("log_target", axis = 1, inplace = True)

foo = train.columns
foo = list(foo)
colNames = foo


# In[ ]:


# Caution! All models and parameter values are just 
# demonstrational and shouldn't be considered as recommended.
# Initialize 1-st level models.
models = [    
    CatBoostRegressor(iterations=200,
                            learning_rate=0.03,
                            depth=4,
                            loss_function='RMSE',
                            eval_metric='RMSE',
                            random_seed=99,
                            od_type='Iter',
                            od_wait=50,
                     logging_level='Silent'),
    
    CatBoostRegressor(iterations=500,
                            learning_rate=0.06,
                            depth=3,
                            loss_function='RMSE',
                            eval_metric='RMSE',
                            random_seed=99,
                            od_type='Iter',
                            od_wait=50,
                     logging_level='Silent'),
    
    ExtraTreesRegressor(random_state = 0, n_jobs = -1, 
        n_estimators = 100, max_depth = 3),
        
    RandomForestRegressor(random_state = 0, n_jobs = -1, 
        n_estimators = 300, max_depth = 3),
    
    XGBRegressor(eta=0.02,reg_lambda=5,reg_alpha=1),
    
    XGBRegressor(eta=0.1,reg_lambda=1,reg_alpha=10),
    
    XGBRegressor(eta=0.02,reg_lambda=1,reg_alpha=10,n_estimators=300),
    
    XGBRegressor(eta=0.012,max_depth=3,n_estimators=200),
    
    GradientBoostingRegressor(),
    
    BaggingRegressor(),
]


# In[ ]:


print(X_train[colNames].shape)
print(X_test[colNames].shape)


# In[ ]:


default_parameters = [0.0229, 408, 3]


# In[ ]:


S_train, S_test = stacking(models, X_train[colNames], y_train, test[colNames], 
    regression = True, metric = mean_absolute_error, n_folds = 4 , 
    shuffle = True, random_state = 0, verbose = 2)


# In[ ]:


print(test.shape)
print(X_train[colNames].shape)
print(S_test.shape)
print(S_train.shape)


# In[ ]:


# Initialize 2-nd level model

model = XGBRegressor(n_estimators=default_parameters[1],
                          learning_rate=default_parameters[0],
                          max_depth=default_parameters[2],
                          random_state=0)
    
# Fit 2-nd level model
model = model.fit(S_train, y_train)


# In[ ]:


# Predict
y_pred = model.predict(S_test)
print(y_pred.shape)


# In[ ]:


result = pd.DataFrame({'ID':test_ID
                       ,'target':np.expm1(y_pred)})

result.head()
result.to_csv('stacked_ensemble_regr_origFeat.csv', index=False)


# In[ ]:


result.head()


# In[ ]:


result.tail()

