#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold,TimeSeriesSplit,KFold,GroupKFold
from sklearn.metrics import roc_auc_score
import sqlite3
import xgboost as xgb
import datetime
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import gc
from sklearn.model_selection import TimeSeriesSplit
import hashlib


# In[ ]:


import os
__print__ = print
def print(string):
    __print__(string)
    os.system(f'echo \"{string}\"')


# In[ ]:


X_train = pd.read_pickle('../input/ieee-cis-12nd-solution-part-1/X_train2.pkl')
X_test = pd.read_pickle('../input/ieee-cis-12nd-solution-part-1/X_test2.pkl')
y_train = pd.read_pickle('../input/ieee-cis-12nd-solution-part-1/y_train2.pkl')
y_train = y_train.isFraud


# In[ ]:


cat = ['uid1','id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9','hour','dow','device_name', 'OS_id_30',  'browser_id_31','ProductID','DeviceInfo__P_emaildomain', 
        'card1__card5', 
        'card2__id_20',
        'card5__P_emaildomain', 
        'addr1__card1',
        'addr1__addr2',
        'card1__card2',
        'card2__addr1',
        'card1__P_emaildomain',
        'card2__P_emaildomain',
        'addr1__P_emaildomain',
        'DeviceInfo__id_31',
        'DeviceInfo__id_20',
        'DeviceType__id_31',
        'DeviceType__id_20',
        'DeviceType__P_emaildomain',
        'card1__M4',
        'card2__M4',
        'addr1__M4',
        'P_emaildomain__M4',
       'uid1__ProductID',
       'uid1__DeviceInfo']


# In[ ]:


cat = set(cat) & set(X_train.columns)


# In[ ]:


for column in cat:
    train_set = set(X_train[column])
    test_set = set(X_test[column])
    tt = train_set.intersection(test_set)
    print('----------------------------------------')
    print(column)
    print(f'train:{len(tt)/len(train_set)}')
    print(f'test:{len(tt)/len(test_set)}')
    X_train[column] = X_train[column].map(lambda x: -999 if x not in tt else x)
    X_test[column] = X_test[column].map(lambda x: -999 if x not in tt else x)


# In[ ]:


kf=KFold(n_splits = 5)
resu1 = 0
impor1 = 0
y_pred = 0
stack_train = np.zeros([X_train.shape[0],])
for train_index, test_index in kf.split(X_train, y_train):
    X_train2= X_train.iloc[train_index,:]
    y_train2= y_train.iloc[train_index]
    X_test2= X_train.iloc[test_index,:]
    y_test2= y_train.iloc[test_index]
    clf = xgb.XGBClassifier(n_estimators=100000, max_depth=11, learning_rate=0.01,random_state=0, subsample=0.8,
                                 colsample_bytree=0.6,min_child_weight = 3,reg_alpha=1,reg_lambda = 0.01,n_jobs=-1,tree_method='gpu_hist')
    clf.fit(X_train2,y_train2,eval_set = [(X_train2,y_train2),(X_test2,y_test2)], eval_metric = 'auc',early_stopping_rounds=500,verbose=30)
    del X_train2,y_train2
    temp_predict = clf.predict_proba(X_test2)[:,1]
    stack_train[test_index] = temp_predict
    y_pred += clf.predict_proba(X_test)[:,1]/5
    roc = roc_auc_score(y_test2, temp_predict)
    print(roc)
    resu1 += roc/5
    impor1 += clf.feature_importances_/5
    del X_test2,y_test2
    gc.collect()
print(f'End:{resu1}')


# In[ ]:


resu = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
resu['isFraud'] = y_pred
resu.to_csv('xgb.csv',index=False)
a= pd.DataFrame()
a['train'] = stack_train
a.to_csv('xgb_train.csv',index=False)

