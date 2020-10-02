#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb


# # Reading the data

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_train.head()


# # Training the data 

# In[ ]:


target = 'winPlacePerc'

train_columns = list(df_train.columns[:-1])

X = df_train[train_columns]
y = df_train[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# In[ ]:


print('Train data', X_train.shape)
print('Train data', df_test.shape)


# In[ ]:


dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
dval= xgb.DMatrix(X_val.values, label=y_val.values)
dtest = xgb.DMatrix(df_test.values)

params = {
    'tree_method': 'hist',
    'objective': "binary:logistic",
    'eval_metric': ['mae'],
    "max_depth" : 30,
    "min_child_weight" : 109,
    "gamma" : 0,
    "reg_alpha" : 1/2**10, "reg_lambda" : 1/2**10, 
    "eta" : 0.05, 
    "subsample" : 1,
    "colsample_bytree" : .8,
    "nthread" : 12,
    "seed" : 1301
}

clf = xgb.train(params, dtrain, evals=[(dtrain, "train"),(dval, 'val')], num_boost_round = 400, early_stopping_rounds=50)


# In[ ]:


df_test['winPlacePerc'] = clf.predict(dtest)

submission = df_test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)

