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
y_train = pd.read_pickle('../input/ieee-cis-12nd-solution-part-1/y_train2.pkl')
y_train = y_train.isFraud


# In[ ]:


## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train = reduce_mem_usage(X_train)')


# In[ ]:


cat = ['uid1', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9','hour','dow','device_name', 'OS_id_30',  'browser_id_31','ProductID',
'DeviceInfo__P_emaildomain', 
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


cat = list(set(cat) & set(X_train.columns))


# In[ ]:


kf=KFold(n_splits = 5)
resu1 = 0
impor1 = 0
y_pred = 0
stack_train = np.zeros([X_train.shape[0],])



for train_index, test_index in kf.split(X_train, y_train):
    
    X_train = pd.read_pickle('../input/ieee-cis-12nd-solution-part-1/X_train2.pkl')
    X_train = reduce_mem_usage(X_train, verbose=False)
    X_train2= X_train.iloc[train_index,:]
    y_train2= y_train.iloc[train_index]
    X_test2= X_train.iloc[test_index,:]
    y_test2= y_train.iloc[test_index]
    
    del X_train
    print('check1')
    clf = cb.CatBoostClassifier(n_estimators=100000, random_state=0, learning_rate= 0.1,depth=10,cat_features = cat,task_type = 'GPU', #learning_rate= 0.05
                               early_stopping_rounds = 400,eval_metric='AUC',border_count = 254,l2_leaf_reg=2)
    clf.fit(X_train2,y_train2,eval_set = (X_test2,y_test2),verbose=100)
    del X_train2,y_train2
    
    print('check2')
    temp_predict = clf.predict_proba(X_test2)[:,1]
    roc = roc_auc_score(y_test2, temp_predict)
    stack_train[test_index] = temp_predict
    print(roc)
    del X_test2,y_test2
    
    print('check3')
    X_test = pd.read_pickle('../input/ieee-cis-12nd-solution-part-1/X_test2.pkl')
    X_test = reduce_mem_usage(X_test, verbose=False)
    y_pred += clf.predict_proba(X_test)[:,1]/5
    del X_test
    
    print('check4')
    resu1 += roc/5
    impor1 += clf.feature_importances_/5
    gc.collect()
print(f'End:{resu1}')


# In[ ]:


resu = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
resu['isFraud'] = y_pred
resu.to_csv('cat.csv',index=False)
a= pd.DataFrame()
a['train'] = stack_train
a.to_csv('cat_train.csv',index=False)

