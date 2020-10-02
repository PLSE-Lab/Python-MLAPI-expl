#!/usr/bin/env python
# coding: utf-8

# I merely use some basic features here. If using some better (but complex) features, I can achieve scores below 0.5 in validate set. However, only ~0.67 score can be obtained on public scoreboard with all these trials.

# In[ ]:


import pandas as pd
import numpy as np
import json
import time
import operator
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor


# In[ ]:


print("\nReading data...")
tn_data = pd.read_json('../input/train.json').set_index('listing_id')
tt_data = pd.read_json('../input/test.json').set_index('listing_id')
print("done")


# In[ ]:


# drop duplicate data
tn_data['features'] = tn_data.loc[:,'features'].apply(lambda x: ' '.join(x))
tn_data['photos'] = tn_data.loc[:,'photos'].apply(lambda x: ' '.join(x))
tt_data['features'] = tt_data.loc[:,'features'].apply(lambda x: ' '.join(x))
tt_data['photos'] = tt_data.loc[:,'photos'].apply(lambda x: ' '.join(x))
tn_data = tn_data.drop_duplicates()


# In[ ]:


# extract needed features to tn_sdata
tn_sdata = tn_data[['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price']]
tt_sdata = tt_data[['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price']]

# define colnames
raw_colnames = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price']


# In[ ]:


# deal with number of images
tn_sdata.loc[:,'n_images'] = tn_data.loc[:,'photos'].apply(lambda x: len(x.split()))
tt_sdata.loc[:,'n_images'] = tt_data.loc[:,'photos'].apply(lambda x: len(x.split()))

# deal with number of features
tn_sdata.loc[:,'n_features'] = tn_data.loc[:,'features'].apply(lambda x: len(x.split()))
tt_sdata.loc[:,'n_features'] = tt_data.loc[:,'features'].apply(lambda x: len(x.split()))

# deal with number of words in descriptions
tn_sdata.loc[:,'n_words'] = tn_data.loc[:,'description'].apply(lambda x: len(x.split()))
tt_sdata.loc[:,'n_words'] = tt_data.loc[:,'description'].apply(lambda x: len(x.split()))

# define colnames
len_colnames = ['n_images', 'n_features', 'n_words']


# In[ ]:


# deal with building ID
tn_sdata.loc[:,'building_id'] = np.array(tn_data.loc[:,'building_id'] != '0', dtype=int).reshape(-1,1)
tt_sdata.loc[:,'building_id'] = np.array(tt_data.loc[:,'building_id'] != '0', dtype=int).reshape(-1,1)
id_colnames = ['building_id']


# In[ ]:


colnames = raw_colnames + len_colnames + id_colnames
print(colnames)


# In[ ]:


# deal with interest_level
tn_cy = tn_data[['interest_level']]
tn_ry = np.zeros((tn_sdata.shape[0],), dtype=int)
tn_ry[np.array(tn_cy['interest_level'] == 'medium')] = 1
tn_ry[np.array(tn_cy['interest_level'] == 'high')] = 1

# merge label into tn_sdata
pd.options.mode.chained_assignment = None  # default='warn'
tn_sdata['interest_level'] = pd.Series(tn_ry, index=tn_sdata.index)


# In[ ]:


# split dataset into subtrain and validate sets
stn_sdata, vld_sdata = train_test_split(tn_sdata, test_size = 0.2, random_state=64)

# partition x and y
stn_x = stn_sdata[colnames]
stn_y = np.array(stn_sdata['interest_level'], dtype=int)
vld_x = vld_sdata[colnames]
vld_y = np.array(vld_sdata['interest_level'], dtype=int)
tt_x = tt_sdata[colnames]


# In[ ]:


clf = XGBRegressor(n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        min_child_weight=1,
                        colsample_bytree=.9,
                        colsample_bylevel=.5,
                        gamma=0.0005,
                        scale_pos_weight=1,
                        base_score=.5,
                        reg_lambda=1,
                        reg_alpha=1,
                        missing=0,
                        seed=514)

k=10
kf = KFold(n_splits=k)
for i, idx in zip(range(k), kf.split(tn_sdata)):
    # split data
    stn_idx, vld_idx = idx
    stn_x = tn_sdata[colnames].iloc[stn_idx]
    vld_x = tn_sdata[colnames].iloc[vld_idx]
    stn_y = np.array(tn_sdata['interest_level'].iloc[stn_idx], dtype=int)
    vld_y = np.array(tn_sdata['interest_level'].iloc[vld_idx], dtype=int)
    
    # train model
    clf.fit(stn_x, stn_y)

    # predict on subtrain and validate sets
    ratio = 0.9
    stn_p = clf.predict(stn_x)
    stn_p = np.array(list(map(lambda x: min(max(x,0),1), stn_p)))
    stn_p = np.array(list(map(lambda x: [1-x, ratio*x, (1-ratio)*x], stn_p)))
    vld_p = clf.predict(vld_x)
    vld_p = np.array(list(map(lambda x: min(max(x,0),1), vld_p)))
    vld_p = np.array(list(map(lambda x: [1-x, ratio*x, (1-ratio)*x], vld_p)))
    
    # prevent -inf caused by 0 prediction
    threshold = 0.0001
    stn_p = stn_p + threshold * np.ones(stn_p.shape)
    stn_p = stn_p / np.sum(stn_p, axis=1).reshape(-1,1)
    vld_p = vld_p + threshold * np.ones(vld_p.shape)
    vld_p = vld_p / np.sum(vld_p, axis=1).reshape(-1,1)
    
    print('\nFold: %d'%i)
    print('Subtrain logloss: %f'%log_loss(stn_y, stn_p, labels=[0,1,2]))
    print('Validate logloss: %f'%log_loss(vld_y, vld_p, labels=[0,1,2]))


# In[ ]:




