#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import lightgbm as lgb


from sklearn.model_selection import train_test_split
import gc
import shap


# This is a basic notebook to show how important each raw feature is according to shap values. I thought that it was worth sharing because it seemed unusual that so many of the values are of approximately equal importance. I don't think that these data are "naturally occuring", rather I think that we are possibly seeing features that have already been engineered by Santander - and perhaps even some principal-component-like features. 

# In[ ]:


data=pd.read_csv('../input/train.csv')

target='target'
feats=[f for f in data.columns if f not in ['target', 'ID_code']]

X_train, X_test, y_train, y_test = train_test_split(data[feats], data[target],
                                                    test_size=0.30,
                                                    random_state=10)


dtrain = lgb.Dataset(data=X_train, 
                     label=y_train, 
                     free_raw_data=False, silent=True)
dvalid = lgb.Dataset(data=X_test, 
                     label=y_test, 
                     free_raw_data=False, silent=True)


params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'nthread': 7,
    'learning_rate': 0.2, 


    'seed': 100,
    'verbose': -1,
    'predict_contrib' : True,
}

clf = lgb.train(
    params=params,
    train_set=dtrain,
    num_boost_round=10000,
    valid_sets=[dtrain, dvalid],
    early_stopping_rounds=200,
    verbose_eval = 100
)


# In[ ]:


shap_values = shap.TreeExplainer(clf).shap_values(data[feats][0:10000])
shap.summary_plot(shap_values, data[feats][0:10000], feature_names=data[feats].columns, max_display=200)

