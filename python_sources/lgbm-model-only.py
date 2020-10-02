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
import gc
print(os.listdir("../input"))


# In[ ]:


import random
seed = 2357
random.seed(seed)


# ## Load Data

# In[ ]:


#Load data
train_df = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


index_f_data = np.load('../input/split-test-dataset/index_of_fake_data.npy')
index_r_data = np.load('../input/split-test-dataset/index_of_real_data.npy')


# In[ ]:


fold_information = pd.read_csv("../input/fold-information/10_fold_information.csv")


# In[ ]:


splits=[]
for i in fold_information.columns:
    a = np.array(list(set(train_df.index)-set(list(fold_information[i].dropna()))))
    a.sort()
    splits.append((np.array(fold_information[i].dropna().astype(int)),a))


# In[ ]:


del fold_information
del a
gc.collect()


# In[ ]:


train_features = train_df.drop(['target','ID_code'], axis = 1)
test_features = test_df.drop(['ID_code'],axis = 1)
train_target = train_df['target']


# In[ ]:


fake_data = test_features.iloc[index_f_data]
real_data = test_features.iloc[index_r_data]


# In[ ]:


train_all_real = pd.concat([train_features,real_data], axis = 0)


# In[ ]:


for f in train_all_real.columns[0:200]:
    train_all_real[f+'duplicate'] = train_all_real.duplicated(f,False).astype(int)


# In[ ]:


for f in train_all_real.columns[0:200]:
    train_all_real[f+'duplicate_count'] = train_all_real.groupby([f])[f].transform('count')/300000
    train_all_real[f+'duplicate_count'] = train_all_real[f+'duplicate_count']*train_all_real[f+'duplicate']


# In[ ]:


for f in train_all_real.columns[0:200]:
    train_all_real[f+'duplicate_value'] = train_all_real[f]*train_all_real[f+'duplicate']


# In[ ]:


train_features_real = train_all_real.iloc[:len(train_target)]
real_data = train_all_real.iloc[len(train_target):len(train_all_real)]


# In[ ]:


train_features_real.shape, real_data.shape


# In[ ]:


del train_all_real
gc.collect()


# In[ ]:


train_all_fake = pd.concat([train_features,fake_data], axis = 0)


# In[ ]:


for f in train_all_fake.columns[0:200]:
    train_all_fake[f+'duplicate'] = train_all_fake.duplicated(f,False).astype(int)


# In[ ]:


for f in train_all_fake.columns[0:200]:
    train_all_fake[f+'duplicate_value'] = train_all_fake[f]*train_all_fake[f+'duplicate']


# In[ ]:


train_features_fake = train_all_fake.iloc[:len(train_target)]
fake_data = train_all_fake.iloc[len(train_target):len(train_all_fake)]


# In[ ]:


del train_all_fake
gc.collect()


# In[ ]:


for f in train_features_real.columns[0:200]:
    train_features_real[f+'distance_of_mean'] = train_features_real[f]-train_features_real[f].mean()
    real_data[f+'distance_of_mean'] = real_data[f]-real_data[f].mean()
    #train_features_fake[f+'distance_of_mean'] = train_features_fake[f]-train_features_fake[f].mean()
    #fake_data[f+'distance_of_mean'] = fake_data[f]-fake_data[f].mean()
    train_features_real[f+'distance_of_mean'] = train_features_real[f+'distance_of_mean']*train_features_real[f+'duplicate']
    real_data[f+'distance_of_mean'] = real_data[f+'distance_of_mean']*real_data[f+'duplicate']
    #train_features_fake[f+'distance_of_mean'] = train_features_fake[f+'distance_of_mean']*train_features_fake[f+'duplicate']
    #fake_data[f+'distance_of_mean'] = fake_data[f+'distance_of_mean']*fake_data[f+'duplicate']


# In[ ]:


train_features_fake.shape,train_features_real.shape,fake_data.shape,real_data.shape,train_target.shape


# In[ ]:


del train_features
del test_features
gc.collect()


# In[ ]:


n_splits = 10# Number of K-fold Splits


# In[ ]:


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.33,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.0085,
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


oof = np.zeros(len(train_features_real))
predictions = np.zeros(len(test_df))
#feature_importance_df = pd.DataFrame()
#features = [c for c in train_features.columns if c not in ['ID_code', 'target']]

for i, (train_idx, valid_idx) in enumerate(splits):  
    print(f'Fold {i + 1}')
    x_train = np.array(train_features_fake)
    y_train = np.array(train_target)
    trn_data = lgb.Dataset(x_train[train_idx.astype(int)], label=y_train[train_idx.astype(int)])
    val_data = lgb.Dataset(x_train[valid_idx.astype(int)], label=y_train[valid_idx.astype(int)])
    
    num_round = 100000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)
    oof[valid_idx] = clf.predict(x_train[valid_idx], num_iteration=clf.best_iteration)
    
    #fold_importance_df = pd.DataFrame()
    #fold_importance_df["feature"] = features
    #fold_importance_df["importance"] = clf.feature_importance()
    #fold_importance_df["fold"] = i + 1
    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions[fake_data.index] += clf.predict(fake_data, num_iteration=clf.best_iteration) / n_splits
    #predictions += clf.predict(test_features, num_iteration=clf.best_iteration) / n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(train_target, oof)))


# In[ ]:


oof = np.zeros(len(train_features_real))
feature_importance_df = pd.DataFrame()
features = [c for c in train_features_real.columns if c not in ['ID_code', 'target']]

for i, (train_idx, valid_idx) in enumerate(splits):  
    print(f'Fold {i + 1}')
    x_train = np.array(train_features_real)
    y_train = np.array(train_target)
    trn_data = lgb.Dataset(x_train[train_idx.astype(int)], label=y_train[train_idx.astype(int)])
    val_data = lgb.Dataset(x_train[valid_idx.astype(int)], label=y_train[valid_idx.astype(int)])
    
    num_round = 100000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)
    oof[valid_idx] = clf.predict(x_train[valid_idx], num_iteration=clf.best_iteration)
    
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions[real_data.index] += clf.predict(real_data, num_iteration=clf.best_iteration) / n_splits
    #predictions += clf.predict(test_features, num_iteration=clf.best_iteration) / n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(train_target, oof)))


# ## Ensemble two model (NN+ LGBM)
# * NN model accuracy is too low, ensemble looks don't work.

# In[ ]:


cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# In[ ]:


#esemble_lgbm_cat = 0.5*oof_cb+0.5*oof
print('LightBGM auc = {:<8.5f}'.format(roc_auc_score(train_target, oof)))
#print('catboost auc = {:<8.5f}'.format(roc_auc_score(train_target, oof_cb)))
#print('LightBGM+catboost auc = {:<8.5f}'.format(roc_auc_score(train_target, esemble_lgbm_cat)))


# In[ ]:


id_code_test = test_df['ID_code']
id_code_train = train_df['ID_code']


# ## Create submit file

# In[ ]:


my_submission_lbgm = pd.DataFrame({"ID_code" : id_code_test, "target" : predictions})
my_submission_train = pd.DataFrame({"ID_code" : id_code_train, "target" : oof})
#my_submission_esemble_lgbm_cat = pd.DataFrame({"ID_code" : id_code_test, "target" : esemble_pred_lgbm_cat})


# In[ ]:


my_submission_lbgm.to_csv('submission_lbgm.csv', index = False, header = True)
my_submission_train.to_csv('submission_lbgm_train.csv', index = False, header = True)
#my_submission_esemble_lgbm_cat.to_csv('my_submission_esemble_lgbm_cat.csv', index = False, header = True)

