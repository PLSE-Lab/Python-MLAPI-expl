#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
import gc
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import plot_importance


# In[ ]:


train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
train = pd.read_csv("../input/train.csv",skiprows=range(1,123903891), nrows=61000000,usecols=train_columns, dtype=dtypes)


# In[ ]:


yTrain = train['is_attributed']
train.drop(['is_attributed'], axis=1, inplace=True)


# In[ ]:


def timeFeatures(df):
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['dow']      = df['datetime'].dt.dayofweek
    df["doy"]      = df["datetime"].dt.dayofyear
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    return df


# In[ ]:


ip_count = train.groupby(['ip'])['channel'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
train = pd.merge(train, ip_count, on='ip', how='left', sort=False)


# In[ ]:


train.drop(['ip'], axis=1, inplace=True)
del ip_count
gc.collect()


# In[ ]:


train = timeFeatures(train)
train.head()


# In[ ]:


# model = RandomForestClassifier()
# model.fit(train,yTrain)
# model.score(train,yTrain)
params = {'eta': 0.3,
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
          'random_state': 99, 
          'silent': True}
dtrain = xgb.DMatrix(train, yTrain)
del train, yTrain
gc.collect()
watchlist = [(dtrain, 'train')]
model = xgb.train(params, dtrain, 30, watchlist, maximize=True, verbose_eval=1)


# In[ ]:


del dtrain


# In[ ]:


test = pd.read_csv("../input/test_supplement.csv",usecols=test_columns, dtype=dtypes)
ip_count = test.groupby(['ip'])['channel'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
test = pd.merge(test, ip_count, on='ip', how='left', sort=False)
test.drop(['ip'], axis=1, inplace=True)
del ip_count
gc.collect()


# In[ ]:


test = timeFeatures(test)
test.head()


# In[ ]:


features = ["app","device","os","channel","clicks_by_ip","dow","doy"]
xTest = test[features]
dtest = xgb.DMatrix(xTest)
pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
del xTest,dtest
gc.collect()


# In[ ]:


my_submission = pd.DataFrame({'click_id': test.click_id, 'is_attributed': pred})


# In[ ]:


my_submission.to_csv('submission.csv', index=False)

