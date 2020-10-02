#!/usr/bin/env python
# coding: utf-8

# **This notebook checks whether the test dataset differs significantly from the train dataset and what to do if it does to avoid getting a completely different evaluation score between the validation and test datasets.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv( '../input/train.csv' )
test = pd.read_csv( '../input/test.csv' )

train = train.drop( ['SalePrice'], axis = 1 )
print(train.shape, test.shape)


# In[ ]:


# Quick Missing Value Imputation
for i in train.columns:
    if train[i].dtype == 'object':
        train[i] = train[i].fillna('None')
    else:
        train[i] = train[i].fillna(-1)
        
for i in test.columns:
    if test[i].dtype == 'object':
        test[i] = test[i].fillna('None')
    else:
        test[i] = test[i].fillna(-1)


# In[ ]:


# One Hot Encoding
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.shape)
print(test.shape)


# In[ ]:


#Which columns are in train but not in test?
print([value for value in train.columns if value not in test.columns])


# In[ ]:


#Let's take the intersection of these columns
cols = [value for value in test.columns if value in train.columns]
train = train[cols]
test = test[cols]
print(train.shape)
print(test.shape)


# In[ ]:


#Adversarial Validation Steps:
train['TARGET'] = 1
test['TARGET'] = 0

data = pd.concat(( train, test ))

x = data.drop( [ 'TARGET', 'Id' ], axis = 1 )
y = data.TARGET

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8 )

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# Now, let's train the model using XGBoost (what else?!)
# Next, we will print the AUC to see the degree of separation between train and test. If AUC is close to 0.5, then that means there is no separation and we're good. If AUC is higher ~0.8-0.9, then there is a high degree of separation and the train and test datasets might differ significantly. In such a case, adversarial validation is useful in selecting a validation dataset which is closest to the test data.

# In[ ]:


xgb_params = {
        'learning_rate': 0.05, 'max_depth': 5,'subsample': 0.9,
        'colsample_bytree': 0.9,'objective': 'binary:logistic',
        'n_estimators':500, 'gamma':1,
        'min_child_weight':1
        }
clf = xgb.XGBClassifier(**xgb_params, seed = 10)

clf.fit(x_train, y_train, eval_set=[(x_test, y_test)],
       eval_metric='auc', verbose=True,early_stopping_rounds=20)

prval = clf.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test,prval))


# So the model cannot distinguish between the train and test dataset. That means we are good to split the dataset between train and validation randomly.
# 
# If the model could distinguish between both datasets that easily, then we could select a validation sample using adversarial validation to make it most representative of the test dataset. The steps for selecting the validation datset would be:
# 
# 1. Score the original train dataset using the XGBoost model
# 2. Sort the predictions by descending order (highest probability first)
# 3. For a 80-20 split, choose the first 20% as validation and the rest as train
