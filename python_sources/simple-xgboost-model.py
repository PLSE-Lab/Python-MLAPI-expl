#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns

import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import xgboost as xgb

import warnings
warnings.simplefilter('ignore')

from scipy import stats
from scipy.stats import norm, skew #for some statistics

import os
print(os.listdir("../input"))


# # Data Acquisition
# Load the data

# In[2]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/train.csv')\ntest = pd.read_csv('../input/test.csv')\n\nprint(train.shape, test.shape)")


# # Data Preparation
# 

# In[3]:


train_id = train['ID_code']
y_train = train['target']
X_train = train.drop(['ID_code', 'target'], axis=1, inplace = False)

test_id = test['ID_code']
X_test = test.drop('ID_code', axis=1, inplace = False)


# # Modelling & Training

# In[4]:


# df_train = xgb.DMatrix(X_train, y_train)
# df_test = xgb.DMatrix(X_test)
# params = {"max_depth":4, "eta":0.3, 'objective': 'binary:logistic'}
# model = xgb.cv(params, df_train,  num_boost_round=10, early_stopping_rounds=5)
model_xgb = xgb.XGBRegressor(n_estimators=5, max_depth=4, learning_rate=0.5) 
model_xgb.fit(X_train, y_train)


# # Evaluation
# #### Using Root Mean Squared Logarithmic Error (RMSLE) evaluation function

# In[6]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

train_pred = model_xgb.predict(X_train)
print('RMSLE : {:.4f}'.format(rmsle(y_train, train_pred)))


# # Prediction 

# In[5]:


xgb_preds = model_xgb.predict(X_test)
solution = pd.DataFrame({"ID_code":test_id, "target":xgb_preds})
solution.to_csv("santander.csv", index = False)
solution.head()


# In[ ]:




