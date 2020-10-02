#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
#reference from excellent notebooks of Bojan,Joao and Pranav
#previous changes give me score of 0.9526
#changes: dropping IP and working on click_time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import time
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


start_time = time.time()
columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }
train=pd.read_csv("../input/train.csv", skiprows=range(1,149903891), nrows=70000000, usecols=columns, dtype=dtypes)
test=pd.read_csv("../input/test.csv")
print('loading of data is completed in [{}] seconds'.format(time.time() - start_time))


# In[ ]:


def datatimeFeatures(df):
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['dow']      = df['datetime'].dt.dayofweek
    df['hod']      = df['datetime'].dt.hour
    df['qoh']      = df['datetime'].dt.quarter
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    return df


# In[ ]:


#handle the categorical data
train = dataPreProcessTime(train)
test = dataPreProcessTime(test)

#select the target variable
y = train['is_attributed']

train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)

# Some feature engineering
nrow_train = train.shape[0]
merge = pd.concat([train, test])

# Make new feature with datatimeFeatures function
merge = datatimeFeatures(merge)

print('preprocessing is completed in [{}] seconds'.format(time.time() - start_time))


# In[ ]:


# Count the number of clicks by ip
ip_count = merge.groupby('ip')['app'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
merge.drop('ip', axis=1, inplace=True)


# In[ ]:


train = merge[:nrow_train]
test = merge[nrow_train:]


# In[ ]:


from sklearn.model_selection import train_test_split
import xgboost as xgb
params = {'eta': 0.6, 
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,         
          'max_depth': 0,             
          'subsample': 0.9,           
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,     
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 84, 
          'silent': True}


# In[ ]:


if (using_test == False):
    # Get 10% of train dataset to use as validation
    x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 300, watchlist, maximize=True, early_stopping_rounds = 50, verbose_eval=10)
else:
    watchlist = [(xgb.DMatrix(train, y), 'train')]
    model = xgb.train(params, xgb.DMatrix(train, y), 15, watchlist, maximize=True, verbose_eval=1)
print('XGBoost Training is finished in [{}] seconds'.format(time.time() - start_time))


# In[ ]:



print('XGBoost Training is finished in [{}] seconds'.format(time.time() - start_time))


# In[ ]:


sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('xgb_submission_hists.csv',index=False)
print('submission is done in [{}] seconds'.format(time.time() - start_time))


# In[ ]:




