#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Seed Everything
import random
seed = 13
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)


# In[ ]:


def read_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


# In[ ]:


# Read the data
train, test = read_data(
    train_path = '../input/house-prices-advanced-regression-techniques/train.csv',
    test_path = '../input/house-prices-advanced-regression-techniques/test.csv',
)


# In[ ]:


def preprocess_data(train, test):
    
    # Concatenate train and test data together
    data = pd.concat([train, test], sort=False)
    
    # Label Encoding
    for f in data.columns:
        if data[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
            
    # Fill in the missing data
    data.fillna(0, inplace=True)

    train = data[: len(train)]
    test = data[-len(test) :]

    return train, test


# In[ ]:


train, test = preprocess_data(train, test)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


y = train["SalePrice"]

# Drop all the ID variables
X = train.drop(["Id", "SalePrice"], axis=1)
X_test = test.drop(["Id", "SalePrice"], axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test2, y_train,  y_test = train_test_split(X, y, test_size=0.3, random_state = 0)


# ### Train a model

# In[ ]:


from sklearn import  linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge


# In[ ]:


# Create linear regression object
# regr = linear_model.LinearRegression()
regr = Ridge(alpha=0.001)

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_val = regr.predict(X_test2)
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean absolute error: %.2f'
      % mean_absolute_error(y_test, y_pred_val))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred_val))


# In[ ]:


submission = pd.DataFrame(
    {
        "Id": test["Id"],
        "SalePrice": y_pred,
    }
)
submission.to_csv("submission_reg.csv", index=False)


# In[ ]:


submission.head()


# In[ ]:


submission['SalePrice'].hist()


# In[ ]:


train['SalePrice'].hist()


# In[ ]:




