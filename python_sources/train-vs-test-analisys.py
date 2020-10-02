#!/usr/bin/env python
# coding: utf-8

# ## waht this notebook do:
# * to predict train or test
# * extract fft feats
# 
# ## what this notebook do not:
# * to predict Power Line Fault
# 
# ## findings
# * high auc ~ 0.99
# * so train and test has different distribution?

# In[ ]:


import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import pyarrow.parquet as pq
import gc
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


smooth = 1000
kaggle_notebook = True


# In[ ]:


if kaggle_notebook:
    read_line = 500
    test_start = 8712
    train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(read_line)]).to_pandas().values.T
    test = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(test_start,test_start+read_line)]).to_pandas().values.T
    meta_tr = pd.read_csv('../input/metadata_train.csv', nrows=read_line)
    meta_te = pd.read_csv('../input/metadata_test.csv', nrows=read_line)
else:
    train = pq.read_pandas('../input/train.parquet').to_pandas().values.T
    test = pq.read_pandas('../input/test.parquet').to_pandas().values.T
    meta_tr = pd.read_csv('../input/metadata_train.csv')
    meta_te = pd.read_csv('../input/metadata_test.csv')


# ### down sampling

# In[ ]:


def smooth_data(arr):
    avarr = np.zeros((arr.shape[0],int(arr.shape[1]/smooth)))
    for i in range(arr.shape[0]):
        for j in range(int(arr.shape[1]/smooth)):
            avarr[i,j] = np.mean(arr[i,smooth*j:smooth*j+smooth-1])
    return avarr
trn = smooth_data(train)
tst = smooth_data(test)


# ### convert by fft

# In[ ]:


def conv_fft(arr):
    farr = np.zeros((arr.shape[0],int(arr.shape[1])))
    for i in range(arr.shape[0]):
        farr[i,:] = np.log(1+np.abs(np.fft.fft(arr[i])))
    return farr
ftrn = conv_fft(trn)
ftst = conv_fft(tst)


# ### add same feats

# In[ ]:


trn_df = pd.DataFrame(ftrn)
tst_df = pd.DataFrame(ftst)

#target
trn_df['is_test'] = 0
tst_df['is_test'] = 1

# additional feats
trn_df['mean'] = np.mean(train,axis=1)
trn_df['max'] = np.max(train,axis=1)
trn_df['min'] = np.min(train,axis=1)
trn_df['phase'] = meta_tr.phase
tst_df['mean'] = np.mean(test,axis=1)
tst_df['max'] = np.max(test,axis=1)
tst_df['min'] = np.min(test,axis=1)
tst_df['phase'] = meta_te.phase

df = pd.concat([trn_df,tst_df])


# ### training

# In[ ]:


params = {'num_leaves': 80,
         'objective':'binary',
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "metric": 'auc'}


# In[ ]:


n_folds = 10
kfolds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=110)
valid_preds = np.zeros(df.shape[0])
for n, (trn_idx, val_idx) in enumerate(kfolds.split(df, df['is_test'])):
    tr = df.iloc[trn_idx].copy()
    va = df.iloc[val_idx].copy()
    tr_y = tr['is_test']
    va_y = va['is_test']
    del tr['is_test'], va['is_test']
    lgtrain = lgb.Dataset(tr, label=tr_y)
    lgvalid = lgb.Dataset(va, label=va_y)
    valid_names=['train','valid']
    valid_sets=[lgtrain, lgvalid]
    bst = lgb.train(params,
                     lgtrain,
                     valid_sets=valid_sets,
                     num_boost_round=100000,
                     early_stopping_rounds=100,
                     verbose_eval=100,
                     )
    valid_preds[val_idx] = bst.predict(va)
score = roc_auc_score(df['is_test'], valid_preds)
print(score)


# In[ ]:


_ =lgb.plot_importance(bst, max_num_features=100, importance_type ='gain', figsize=(15,50))


# In[ ]:


meta_tr['pred'] = valid_preds[:len(trn_df)]
meta_tr.to_csv('metadata_train_for_AdversarialValidation.csv', index=False)


# ### Let's check whether we get the same result, if we split test to test1 and test2 and try to predict if it is test1 or test2?

# In[ ]:


del train, test, trn, tst, ftrn, ftst
gc.collect()
read_line = 1000
test_start = 8712
test = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(test_start,test_start+read_line)]).to_pandas().values.T
meta_te = pd.read_csv('../input/metadata_test.csv', nrows=read_line)
train, test, meta_tr, meta_te = train_test_split(test, meta_te, test_size=0.5, random_state=1)


# In[ ]:


trn = smooth_data(train)
tst = smooth_data(test)
ftrn = conv_fft(trn)
ftst = conv_fft(tst)
trn_df = pd.DataFrame(ftrn)
tst_df = pd.DataFrame(ftst)
trn_df['is_test'] = 0
tst_df['is_test'] = 1
trn_df['mean'] = np.mean(train,axis=1)
trn_df['max'] = np.max(train,axis=1)
trn_df['min'] = np.min(train,axis=1)
trn_df['phase'] = meta_tr.phase.values
tst_df['mean'] = np.mean(test,axis=1)
tst_df['max'] = np.max(test,axis=1)
tst_df['min'] = np.min(test,axis=1)
tst_df['phase'] = meta_te.phase.values
df = pd.concat([trn_df,tst_df])

params = {'num_leaves': 80,
         'objective':'binary',
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "metric": 'auc'}

n_folds = 10
kfolds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=110)
valid_preds = np.zeros(df.shape[0])
for n, (trn_idx, val_idx) in enumerate(kfolds.split(df, df['is_test'])):
    tr = df.iloc[trn_idx].copy()
    va = df.iloc[val_idx].copy()
    tr_y = tr['is_test']
    va_y = va['is_test']
    del tr['is_test'], va['is_test']
    lgtrain = lgb.Dataset(tr, label=tr_y)
    lgvalid = lgb.Dataset(va, label=va_y)
    valid_names=['train','valid']
    valid_sets=[lgtrain, lgvalid]
    bst = lgb.train(params,
                     lgtrain,
                     valid_sets=valid_sets,
                     num_boost_round=100000,
                     early_stopping_rounds=100,
                     verbose_eval=100,
                     )
    valid_preds[val_idx] = bst.predict(va)
score = roc_auc_score(df['is_test'], valid_preds)
print(score)


# In[ ]:


_ =lgb.plot_importance(bst, max_num_features=100, importance_type ='gain', figsize=(15,50))

