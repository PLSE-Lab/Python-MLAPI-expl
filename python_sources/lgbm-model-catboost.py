#!/usr/bin/env python
# coding: utf-8

# ## Pytorch to implement simple feed-forward NN model (0.89+)
# 
# * As below discussion, NN model can get lB 0.89+
# * https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/82499#latest-483679
# * Add Cycling learning rate , K-fold cross validation (0.85 to 0.86)
# * Add flatten layer as below discussion (0.86 to 0.897)
# * https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/82863
# 
# ## LightGBM (LB 0.899)
# 
# * Fine tune parameters (0.898 to 0.899)
# * Reference this kernel : https://www.kaggle.com/chocozzz/santander-lightgbm-baseline-lb-0-899
# 
# 
# ## Plan to do
# * Modify model structure on NN model
# * Focal loss
# * Feature engineering
# * Tune parameters oof LightGBM

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import os
print(os.listdir("../input"))
import gc


# ## Load Data

# In[ ]:


#Load data
train_df = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


## Use no scaling data to train LGBM
train_features = train_df.drop(['target','ID_code'], axis = 1)
test_features = test_df.drop(['ID_code'],axis = 1)
train_target = train_df['target']


# In[ ]:


gc.collect()


# In[ ]:


train_all = pd.concat((train_features,test_features),axis = 0)


# In[ ]:


for f in train_all.columns:
    train_all[f+'_duplicate'] = train_all.duplicated(f,False).astype(int)
#train_all['count_total_all']=train_all.iloc[:,200:400].sum(axis=1)


# In[ ]:


for f in train_all.columns[0:200]:
    train_all[f+'duplicate_value'] = train_all[f]*train_all[f+'_duplicate']


# In[ ]:


train_features = train_all.iloc[:200000]
test_features = train_all.iloc[200000:400000]


# In[ ]:


del train_all
gc.collect()


# In[ ]:


#test_features['var_68_te'].value_counts()


# In[ ]:


train_features.shape, test_features.shape, train_target.shape


# In[ ]:


n_splits = 7# Number of K-fold Splits

splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True).split(train_features, train_target))
splits[:3]


# In[ ]:


cat_params = {
    'learning_rate':0.01,
    'max_depth':2,
    'eval_metric': 'AUC',
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 1,
    'objective': 'Logloss',
    'od_type': 'Iter',
    'l2_leaf_reg': 2,
    'allow_writing_files': False}


# In[ ]:


from catboost import CatBoostClassifier
oof_cb = np.zeros(len(train_features))
predictions_cb = np.zeros(len(test_features))
#feature_importance_df = pd.DataFrame()
#features = [c for c in train_features_df.columns if c not in ['ID_code', 'target']]

for i, (train_idx, valid_idx) in enumerate(splits):  
    print(f'Fold {i + 1}')
    x_train = np.array(train_features)
    y_train = np.array(train_target)
    trn_x = x_train[train_idx.astype(int)]
    trn_y = y_train[train_idx.astype(int)]
    val_x = x_train[valid_idx.astype(int)]
    val_y = y_train[valid_idx.astype(int)]
    
    num_round = 100000
    clf = CatBoostClassifier( num_round, task_type='GPU', early_stopping_rounds=1000,**cat_params,)
    clf.fit(trn_x, trn_y, eval_set=(val_x, val_y), cat_features=[], use_best_model=True, verbose=500)
    
    oof_cb[valid_idx] = clf.predict_proba(val_x)[:,1]

    predictions_cb += clf.predict_proba(test_features)[:,1] / 5

print("CV score: {:<8.5f}".format(roc_auc_score(train_target, oof_cb)))
gc.collect()


# In[ ]:


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.33,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 12,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': 1
}


# In[ ]:


oof = np.zeros(len(train_features))
predictions = np.zeros(len(test_features))
#feature_importance_df = pd.DataFrame()
#features = [c for c in train_features.columns if c not in ['ID_code', 'target']]

for i, (train_idx, valid_idx) in enumerate(splits):  
    print(f'Fold {i + 1}')
    x_train = np.array(train_features)
    y_train = np.array(train_target)
    trn_data = lgb.Dataset(x_train[train_idx.astype(int)], label=y_train[train_idx.astype(int)])
    val_data = lgb.Dataset(x_train[valid_idx.astype(int)], label=y_train[valid_idx.astype(int)])
    
    num_round = 100000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[valid_idx] = clf.predict(x_train[valid_idx], num_iteration=clf.best_iteration)
    
    #fold_importance_df = pd.DataFrame()
    #fold_importance_df["feature"] = features
    #fold_importance_df["importance"] = clf.feature_importance()
    #fold_importance_df["fold"] = i + 1
    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_features, num_iteration=clf.best_iteration) / 5

print("CV score: {:<8.5f}".format(roc_auc_score(train_target, oof)))
gc.collect()


# ## Ensemble two model (NN+ LGBM)
# * NN model accuracy is too low, ensemble looks don't work.

# In[ ]:


esemble_lgbm_cat = 0.5*oof_cb+0.5*oof
print('LightBGM auc = {:<8.5f}'.format(roc_auc_score(train_target, oof)))
print('catboost auc = {:<8.5f}'.format(roc_auc_score(train_target, oof_cb)))
print('LightBGM+catboost auc = {:<8.5f}'.format(roc_auc_score(train_target, esemble_lgbm_cat)))


# In[ ]:


esemble_pred_lgbm_cat = 0.5*predictions+0.5*predictions_cb


# In[ ]:


id_code_test = test_df['ID_code']


# ## Create submit file

# In[ ]:


my_submission_lbgm = pd.DataFrame({"ID_code" : id_code_test, "target" : predictions})
my_submission_cat = pd.DataFrame({"ID_code" : id_code_test, "target" : predictions_cb})
my_submission_esemble_lgbm_cat = pd.DataFrame({"ID_code" : id_code_test, "target" : esemble_pred_lgbm_cat})


# In[ ]:


my_submission_lbgm.to_csv('submission_lbgm.csv', index = False, header = True)
my_submission_cat.to_csv('submission_cb.csv', index = False, header = True)
my_submission_esemble_lgbm_cat.to_csv('my_submission_esemble_lgbm_cat.csv', index = False, header = True)


# In[ ]:




