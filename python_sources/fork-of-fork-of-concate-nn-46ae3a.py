#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.optimizer import Optimizer

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


#Load data
train_df = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')


# In[ ]:


magic_index = pd.read_csv('../input/magicdddd/78734.csv',header = None)


# In[ ]:


magic_index_list = list(magic_index[0])


# In[ ]:


magic_valid = list(set(train_df.index)-set(magic_index_list))


# In[ ]:


train_features = train_df.drop(['target','ID_code'], axis = 1)
test_features = test_df.drop(['ID_code'],axis = 1)
train_target = train_df['target']


# In[ ]:


large_duplicate_features=['var_5','var_10','var_11','var_17','var_18','var_19','var_20','var_21','var_26','var_30','var_40','var_41','var_44','var_45','var_47','var_48','var_49','var_51','var_54','var_55','var_61','var_67','var_70','var_73','var_74','var_75','var_76','var_80','var_82','var_83','var_84','var_86','var_87','var_90','var_96','var_97','var_100','var_102','var_107','var_117','var_118','var_120','var_123','var_134','var_135','var_136','var_137','var_139','var_141','var_142','var_147','var_149','var_155','var_157','var_158','var_160','var_167','var_171','var_172','var_173','var_174','var_176','var_178','var_182','var_184','var_187','var_199']


# In[ ]:


less_duplicate_features =['var_4']


# In[ ]:


train_df_copy = train_df.copy()
test_df_copy = test_df.copy()
for col in train_df.columns[2:202]:
    duplicate = list(set(list(train_features[col].value_counts().index)) & set(list(test_features[col].value_counts().index)))
    train_df_copy[col+'train_test_have'] = train_df_copy[col].isin(duplicate).astype(int)
    test_df_copy[col+'train_test_have'] = test_df_copy[col].isin(duplicate).astype(int)


# In[ ]:


for f in train_df.columns[2:202]:
    train_df_copy[f+'duplicate_value'] = round(train_df_copy[f],4)*train_df_copy[f+'train_test_have']
    test_df_copy[f+'duplicate_value'] = round(test_df_copy[f],4)*test_df_copy[f+'train_test_have']


# In[ ]:


train_features = train_df_copy.drop(['target','ID_code'], axis = 1)
test_features = test_df_copy.drop(['ID_code'],axis = 1)
train_target = train_df_copy['target']


# In[ ]:


train_all = pd.concat([train_features,test_features], axis = 0)


# In[ ]:


for f in train_all.columns[0:200]:
    train_all[f+'duplicate'] = ((train_all.duplicated(f,False).astype(int))-1)*(-1)
    #train_all[f+'duplicate_value2'] = train_all[f] + train_all[f+'duplicate']


# In[ ]:


"""
for f in train_all.columns[0:200]:
    train_df_copy[f+'duplicate'] = ((train_df_copy.duplicated(f,False).astype(int))-1)*(-1)
    test_df_copy[f+'duplicate'] = ((test_df_copy.duplicated(f,False).astype(int))-1)*(-1)
    train_df_copy[f+'duplicate_value2'] = train_df_copy[f] + train_df_copy[f+'duplicate']
    test_df_copy[f+'duplicate_value2'] = test_df_copy[f] + test_df_copy[f+'duplicate']
"""


# In[ ]:


train_df_copy.shape,test_df_copy.shape


# In[ ]:


"""
train_features = train_df_copy.drop(['target','ID_code'], axis = 1)
test_features = test_df_copy.drop(['ID_code'],axis = 1)
train_target = train_df_copy['target']
"""


# In[ ]:


train_features = train_all.iloc[:len(train_target)]
test_features = train_all.iloc[len(train_target):len(train_all)]


# In[ ]:


gc.collect()


# In[ ]:


train_valid = list(set(train_features.index)-set(magic_valid))


# In[ ]:


gc.collect()


# In[ ]:


train_features_array = np.array(train_features)
train_target_array = np.array(train_target)
x_train = train_features_array[np.array(train_valid)]
y_train = train_target_array[np.array(train_valid)]
x_valid = train_features_array[np.array(magic_valid)]
y_valid = train_target_array[np.array(magic_valid)]


# In[ ]:


#x_train,x_valid,y_train,y_valid = train_test_split(train_features,train_target, test_size=0.2, stratify=train_target, random_state=0)


# In[ ]:


del train_features_array
del train_target_array
gc.collect()


# In[ ]:


x_train.shape,x_valid.shape


# In[ ]:


del train_df_copy
del test_df_copy
del train_all
gc.collect()


# In[ ]:


oof_cb = np.zeros(len(train_features))
predictions_cb = np.zeros(len(test_df))

from catboost import CatBoostClassifier
cat_params = {
    'learning_rate':0.01,
    'max_depth':3,
    'eval_metric': 'AUC',
    'bootstrap_type': 'Bernoulli',
    'objective': 'Logloss',
    'od_type': 'Iter',
    'l2_leaf_reg': 13,
    'random_seed': 2500,
    'allow_writing_files': False}

num_round = 50000
clf = CatBoostClassifier( num_round, task_type='GPU', early_stopping_rounds=3000,**cat_params)
clf.fit(x_train, y_train, eval_set=(x_valid, y_valid), cat_features=[], use_best_model=True, verbose=500)

predictions_cb = clf.predict_proba(test_features)[:,1]
valid_cb = clf.predict_proba(x_valid)[:,1]
oof_cb = clf.predict_proba(x_train)[:,1]
print("CV score: {:<8.5f}".format(roc_auc_score(y_valid, valid_cb)))
gc.collect()


# In[ ]:


feature_importance_df = pd.DataFrame()
features = [c for c in train_features.columns if c not in ['ID_code', 'target']]
param = {
        'bagging_freq': 5,
        'bagging_fraction': 0.4,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.05,
        'learning_rate': 0.01,
        'max_depth': -1,  
        'metric':'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 11,
        'tree_learner': 'serial',
        'objective': 'binary', 
        'verbosity': 1
    }


trn_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_valid, label=y_valid)
oof = np.zeros(len(x_train))
predictions = np.zeros(len(test_features))

num_round = 100000
clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
oof = clf.predict(x_train, num_iteration=clf.best_iteration)
valid = clf.predict(x_valid, num_iteration=clf.best_iteration)
predictions = clf.predict(test_features, num_iteration=clf.best_iteration)

fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = features
fold_importance_df["importance"] = clf.feature_importance()
feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

print("CV score: {:<8.5f}".format(roc_auc_score(y_valid, valid)))
gc.collect()


# In[ ]:


feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)
plt.figure(figsize=(10,50))
sns.barplot('importance', 'feature', data=feature_importance_df)


# In[ ]:


test = pd.read_csv('../input/ensemble-usesssssss/submission_ensemble.csv')
test2 = pd.read_csv('../input/ensemble-use/submission_ensemble.csv')


# In[ ]:


id_code_test = test_df['ID_code']


# In[ ]:


my_submission_lbgm = pd.DataFrame({"ID_code" : id_code_test, "target" : predictions})


# In[ ]:


from scipy.stats.stats import pearsonr  
print(pearsonr(test.target.rank(),my_submission_lbgm.target.rank()))
print(pearsonr(test2.target.rank(),my_submission_lbgm.target.rank()))


# In[ ]:


#fold_importance_df = fold_importance_df.sort_values(by="importance",ascending=False)


# In[ ]:


#esemble_nn_lgbm = 0.8*valid + 0.2* valid_preds
#esemble_nn_cat = 0.6*valid_cb + 0.4* valid_preds
esemble_lgbm_cat = 0.5*valid_cb+0.5*valid
#esemble_all = 0.4*valid_cb + 0.4*valid + 0.2*valid_preds
#print('NN auc = {:<8.5f}'.format(roc_auc_score(y_valid, valid_preds)))
print('LightBGM auc = {:<8.5f}'.format(roc_auc_score(y_valid, valid)))
print('catboost auc = {:<8.5f}'.format(roc_auc_score(y_valid, valid_cb)))
#print('NN+LightBGM auc = {:<8.5f}'.format(roc_auc_score(y_valid, esemble_nn_lgbm)))
#print('NN+catboost auc = {:<8.5f}'.format(roc_auc_score(y_valid, esemble_nn_cat)))
print('LightBGM+catboost auc = {:<8.5f}'.format(roc_auc_score(y_valid, esemble_lgbm_cat)))
#print('All = {:<8.5f}'.format(roc_auc_score(y_valid, esemble_all)))


# In[ ]:


id_code_test = test_df['ID_code']


# In[ ]:


#esemble_pred_nn_lgbm = 0.2* test_preds+ 0.8 *predictions
#esemble_pred_nn_cat = 0.4* test_preds+ 0.6 *predictions_cb
esemble_pred_lgbm_cat = 0.5*predictions+0.5*predictions_cb
#esemble_all = 0.4*predictions + 0.4*predictions_cb + 0.3*test_preds


# In[ ]:


#my_submission_nn = pd.DataFrame({"ID_code" : id_code_test, "target" : test_preds})
my_submission_lbgm = pd.DataFrame({"ID_code" : id_code_test, "target" : predictions})
my_submission_cat = pd.DataFrame({"ID_code" : id_code_test, "target" : predictions_cb})
#my_submission_esemble_nn_lgbm = pd.DataFrame({"ID_code" : id_code_test, "target" : esemble_pred_nn_lgbm})
#my_submission_esemble_nn_cat = pd.DataFrame({"ID_code" : id_code_test, "target" : esemble_pred_nn_cat})
my_submission_esemble_lgbm_cat = pd.DataFrame({"ID_code" : id_code_test, "target" : esemble_pred_lgbm_cat})
#my_submission_esemble_all = pd.DataFrame({"ID_code" : id_code_test, "target" : esemble_all})


# In[ ]:


#my_submission_nn.to_csv('submission_nn.csv', index = False, header = True)
my_submission_lbgm.to_csv('submission_lbgm.csv', index = False, header = True)
my_submission_cat.to_csv('submission_cb.csv', index = False, header = True)
#my_submission_esemble_nn_lgbm.to_csv('my_submission_esemble_nn_lgbm.csv', index = False, header = True)
#my_submission_esemble_nn_cat.to_csv('my_submission_esemble_nn_cat.csv', index = False, header = True)
my_submission_esemble_lgbm_cat.to_csv('my_submission_esemble_lgbm_cat.csv', index = False, header = True)
#my_submission_esemble_all.to_csv('my_submission_esemble_all.csv', index = False, header = True)


# In[ ]:


my_submission_esemble_lgbm_cat.head()


# In[ ]:




