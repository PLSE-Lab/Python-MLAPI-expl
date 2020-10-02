#!/usr/bin/env python
# coding: utf-8

# # LightGBM
# Simple, short kernel - just pure LightGBM ( no SKF)
# 
# PS
# 
# Params for lgb taken from kernel: https://www.kaggle.com/vincentlugat/skf-lightgbm-target-encoding "SKF LightGBM - Target Encoding"
# 
# Here is similar kernel for the second "cat in dat" competition:
# https://www.kaggle.com/alexandervc/lightgbm
# Current outcome is similar - 0.005 roc_auc_ from the current to solution
# 

# In[ ]:





# # Data Load 

# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
import time 
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split 
import datetime

print('Load starts.', datetime.datetime.now() )

s0 = time.time()
train = pd.read_csv("../input/cat-in-the-dat/train.csv")#, index_col="id")
test = pd.read_csv("../input/cat-in-the-dat/test.csv")# , index_col="id")
print(train.shape, test.shape)
print(train.head(2))
print(test.head(2))

X = train.drop(['id','target'], axis = 1)
categorical_features = [col for c, col in enumerate(X.columns)                         if not ( np.issubdtype(X.dtypes[c], np.number )  )  ]
y = train['target']

print( len(categorical_features), X.shape, y.shape, y.mean()  )
for f in categorical_features:
    X[f] = X[f].astype('category')

X1,X2, y1,y2 = train_test_split(X,y, test_size = 0.2, random_state = 0, stratify = y )
print(X1.shape, X2.shape, y1.shape, y2.shape, y1.mean(), y2.mean(), y.mean() )

print(time.time() - s0)
print('Load finished.', datetime.datetime.now() )


# # LightGBM model

# In[ ]:


model = lgb.LGBMClassifier(**{
                'learning_rate': 0.05,
                'feature_fraction': 0.1,
                'min_data_in_leaf' : 12,
                'max_depth': 3,
                'reg_alpha': 1,
                'reg_lambda': 1,
                'objective': 'binary',
                'metric': 'auc',
                'n_jobs': -1,
                'n_estimators' : 5000,
                'feature_fraction_seed': 42,
                'bagging_seed': 42,
                'boosting_type': 'gbdt',
                'verbose': 1,
                'is_unbalance': True,
                'boost_from_average': False})

import datetime
print('Start fit.', datetime.datetime.now() )

model = model.fit(X1, y1,
                  eval_set = [(X1, y1), 
                              (X2, y2)],
                  verbose = 1,
                  eval_metric = 'auc',
                  early_stopping_rounds = 1000)

print('End fit.', datetime.datetime.now() )


# # PRED CSV

# In[ ]:


X_test = test.drop('id',axis = 1 )
for f in categorical_features:
    X_test[f] = X_test[f].astype('category')
    
pd.DataFrame({'id': test['id'], 'target': model.predict_proba(X_test)[:,1]}).to_csv('submission.csv', index=False)


# In[ ]:




