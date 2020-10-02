#!/usr/bin/env python
# coding: utf-8

# # Overview of brilliant kernels 

# ## 1.data preprocessing & FE
# ### downsample of train data

# 
# * one-hot encoding/mean encoding for app,channel
# * time pattern analysis
# * keep all positive observations (observations with is_attributed = 1),randomly selected 5 million observations from null observations(observations with is_attributed = 1)

# ### data observations
# * most frequent hours in test data:[4,5,9,10,13,14]
# * least frequent hours in test data:[6,11,15]

# ### new created features
# * treat **ip,device,os** as an unique user instance
# * find discriminative features from combination of (ip,device,os) and app, for they are independent of the ad publishers

# ### integrated features
# * ip-wday-hour
# * ip-wday-hour-os
# * ip-wday-hour-app
# * ip-wday-hour-app-os
# * app-wday-hour

# ### experiments
# * most previous kernels incorporate FE that combina multi given features, but intuitively, some of them make no sense,so i propose to use simplified given features to train the classification model.
# * the ip pattern analysis indicates that **ip** feature could be divided into 4 periods, so I just chose training set that has ip>=126420,leading a local 0.979 score.

# In[ ]:


import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb

def dataPreProcessTime(df):
    df['click_time'] = pd.to_datetime(df['click_time']).dt.date
    df['click_time'] = df['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    
    return df

start_time = time.time()

train = pd.read_csv('../input/train.csv', skiprows=160000000, nrows=40000000)
test = pd.read_csv('../input/test.csv')
train.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']

print('[{}] Finished loading data'.format(time.time() - start_time))

train = dataPreProcessTime(train)
test = dataPreProcessTime(test)

train=train[train.ip>=126420]

y = train['is_attributed']
train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)


# train.drop('click_time',axis=1,inplace=True)
# train.drop('os',axis=1,inplace=True)
# train.drop('ip',axis=1,inplace=True)
# train.drop('app',axis=1,inplace=True)


print('[{}] Start XGBoost Training'.format(time.time() - start_time))

params = {'eta': 0.1, 
          'max_depth': 4, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':4,
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'random_state': 99, 
          'silent': True,
          'subsample':0.5}
          
x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 260, watchlist, maximize=True, verbose_eval=10)

print('[{}] Finish XGBoost Training'.format(time.time() - start_time))

sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('syp_xgb_sub_ip_le.csv',index=False)


# ## 3.params tuning for xgb
# 

# ### tips for tuning params

# **max_depth** : 
# *    higher tree depth makes more complicated models, also making it possible to learn more local and concrete samples. 
# *    it often be set during 3~10
# *    the cross validation could help to tune tree depth
# 
# **filter features from fea importance generated from xgb**
# 
# **little learning rate with larger n_estimators that could get from CV**
# 
# **subsample,colsample_bytree,colsample_bylevel could be set in 0.3-0.8, random sample is better in most conditions**

# ## 4.blending&stacking
# * with weight
# * logit combination
# * average 
# * rank

# ## references
#  i wrote this gratefully with reference to most highly upvoted kernes

# In[ ]:




