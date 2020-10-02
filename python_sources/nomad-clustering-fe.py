#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

import os
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/nomad-feature-engineering/train.csv')
test = pd.read_csv('../input/nomad-feature-engineering/test.csv')

train_clus = pd.read_csv('../input/nomad-cluster/trainclusterlabels.tsv', header=None, names=['cluster'])
test_clus = pd.read_csv('../input/nomad-cluster/testclusterlabels.tsv', header=None, names=['cluster'])

train = pd.merge(train, train_clus, left_index=True, right_index=True)
test = pd.merge(test, test_clus, left_index=True, right_index=True)


# In[ ]:


categorical = ['spacegroup', 'number_of_total_atoms', "cluster"
               'a1', 'a2', 'a3', 'a4', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15',
               'SG_12', 'SG_33', 'SG_167', 'SG_194', 'SG_206', 'SG_227']

for c in train.columns:
    if c in categorical:
        train[c] = train[c].astype('category')
    else:
        train[c] = train[c].astype('float64')
        
for c in test.columns:
    if c in categorical:
        test[c] = test[c].astype('category')
    else:
        test[c] = test[c].astype('float64')


# In[ ]:


formation = train['formation_energy_ev_natom']
bandgap = train['bandgap_energy_ev']


# In[ ]:


param = {
    'num_leaves': 7,
    'objective': 'regression',
    'min_data_in_leaf': 18,
    'learning_rate': 0.04,
    'feature_fraction': 0.93,
    'bagging_fraction': 0.93,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 1
}


# In[ ]:


num_folds = 11
features = [c for c in train.columns if c not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]]

folds = KFold(n_splits=num_folds, random_state=2319)
getVal1 = np.zeros(len(train))
predictions1 = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

print('Light GBM Model')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, formation.values)):
    
    X_tr, y_tr = train.iloc[trn_idx][features], formation.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], formation.iloc[val_idx]
    
    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=formation.iloc[val_idx])
    
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    
    getVal1[val_idx] += clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions1 += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV RMSLE: {:<8.5f}".format(np.sqrt(mse(formation, getVal1))))


# In[ ]:


folds = KFold(n_splits=num_folds, random_state=2319)
getVal2 = np.zeros(len(train))
predictions2 = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

print('Light GBM Model')
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, bandgap.values)):
    
    X_tr, y_tr = train.iloc[trn_idx][features], bandgap.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx][features], bandgap.iloc[val_idx]
    
    print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=bandgap.iloc[val_idx])
    
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    
    getVal2[val_idx] += clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions2 += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV RMSLE: {:<8.5f}".format(np.sqrt(mse(bandgap, getVal2))))


# In[ ]:


sub = pd.read_csv('../input/nomad-feature-engineering/test.csv')
features = [c for c in train.columns if c not in ["id", "formation_energy_ev_natom", "bandgap_energy_ev", "cluster"]]
sub = sub.drop(features, axis=1)

sub["formation_energy_ev_natom"] = np.exp(predictions1) - 1
sub["bandgap_energy_ev"] = np.exp(predictions2) - 1
sub.to_csv('submission.csv', index=False)


train = pd.read_csv('../input/nomad-feature-engineering/train.csv')

train["predicted_fe"] = getVal1
train["predicted_be"] = getVal2
train.to_csv('train.csv', index=False)


test = pd.read_csv('../input/nomad-feature-engineering/test.csv')

test["predicted_fe"] = predictions1
test["predicted_be"] = predictions2
test.to_csv('test.csv', index=False)

