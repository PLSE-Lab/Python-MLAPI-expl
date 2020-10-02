#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from collections import Counter
from contextlib import contextmanager
import gc
import os
import psutil
import time
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
#from sklearn.preprocessing import StandardScaler
#from tsfresh.feature_extraction import feature_calculators
from tqdm import tqdm_notebook as tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import random as rn
import scipy as sp
import itertools
import warnings
import pywt
import math
import warnings
warnings.filterwarnings("ignore")
from scipy import signal

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.model_selection import check_cv

from category_encoders import CatBoostEncoder

def reduce_mem_usage(df,verbose=True):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min
                        and c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min
                      and c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min
                      and c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min
                      and c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min
                      and c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    msg = f'Mem. usage decreased to {end_mem:5.2f} MB ({reduction * 100:.1f} % reduction)'
    if verbose:
        print(msg)

    return df


# In[ ]:


submission  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv') 
train = pd.read_csv('/kaggle/input/remove-trends-giba/train_clean_giba.csv')
test  = pd.read_csv('/kaggle/input/remove-trends-giba/test_clean_giba.csv')
del train["type"], test["type"]

gc.collect()

train["category"] = 0
test["category"] = 0

# train segments with more then 9 open channels classes
train.loc[2_000_000:2_500_000, 'category'] = 1
train.loc[4_500_000:5_000_000, 'category'] = 1

# test segments with more then 9 open channels classes (potentially)
test.loc[500_000:600_000, "category"] = 1
test.loc[700_000:800_000, "category"] = 1

train['group'] = np.arange(train.shape[0])//500_000    
aug_df = train[train["group"] == 5].copy()
aug_df["group"] = 10
aug_df["category"] = 1
for col in ["signal", "open_channels"]:
    aug_df[col] += train[train["group"] == 8][col].values

train = train.append(aug_df, sort=False).reset_index(drop=True)
del aug_df
gc.collect()

y=train['open_channels']

# train = reduce_mem_usage(train,verbose=True)
# test = reduce_mem_usage(test,verbose=True)

print(train.shape)


# In[ ]:


train['batch'] = np.arange(train.shape[0])//100_000
test['batch'] = np.arange(test.shape[0])//100_000

shift_sizes = np.arange(1,21)
for temp in [train,test]:
    for shift_size in shift_sizes:    
        temp['signal_shift_pos_'+str(shift_size)] = temp.groupby('batch')['signal'].shift(shift_size).fillna(-3)
        # temp['signal_shift_pos_'+str(shift_size)] = temp.groupby("batch")['signal_shift_pos_'+str(shift_size)].transform(lambda x: x.bfill())
        temp['signal_shift_neg_'+str(shift_size)] = temp.groupby('batch')['signal'].shift(-1*shift_size).fillna(-3)
        # temp['signal_shift_neg_'+str(shift_size)] = temp.groupby("batch")['signal_shift_neg_'+str(shift_size)].transform(lambda x: x.ffill())


# In[ ]:


remove_fea=['time','batch','batch_index','batch_slices','batch_slices2','group',"open_channels"]
features=[i for i in train.columns if i not in remove_fea]
print(len(features))
train[features].head()


# In[ ]:


n_splits=5    
cv_result = []
cv_pred = []
oof_preds = np.zeros(train.shape[0])
y_preds = np.zeros(test.shape[0])

target = "open_channels"
train['group'] = np.arange(train.shape[0])//4000
group = train['group']
kf = GroupKFold(n_splits=5)
splits = [x for x in kf.split(train, y, group)]

for fold, (tr_ind, val_ind) in enumerate(splits):
    x_train, x_val = train[features].iloc[tr_ind], train[features].iloc[val_ind]
    y_train, y_val = y[tr_ind], y[val_ind]
    print(f'Fold {fold + 1}, {x_train.shape}, {x_val.shape}')
    train_set = lgb.Dataset(x_train, y_train)
    val_set = lgb.Dataset(x_val, y_val)

    params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'bagging_fraction': 0.75,#0.75
                'bagging_freq': 5,      #5
                'learning_rate': 0.05,
                'feature_fraction': 0.5,
                'max_depth': -1,
                'num_leaves': 64,      #64
                "reg_alpha": 0.1,
                "reg_lambda": 10,       #10
                'verbose': -1,
                'seed':2019
                }

    model = lgb.train(params, train_set, num_boost_round = 3500, early_stopping_rounds =100, valid_sets=[train_set, val_set],verbose_eval = 100)
    del x_train,y_train
    gc.collect()
    oof_preds[val_ind] = model.predict(x_val,num_iteration=model.best_iteration)
    del x_val
    gc.collect()
    result = f1_score(y_val,np.round(np.clip(oof_preds[val_ind], 0, 10)).astype(int),average='macro')
    print('f1 score : ',result)
    cv_result.append(round(result,5))
    y_preds += model.predict(test[features],num_iteration=model.best_iteration)/n_splits


# In[ ]:


f1_score(y,np.round(np.clip(oof_preds, 0, 10)).astype(int),average='macro')


# In[ ]:


f1_score(y[:5000_000],np.round(np.clip(oof_preds[:5000_000], 0, 10)).astype(int),average='macro')


# In[ ]:


train["oof"] = np.round(np.clip(oof_preds, 0, 10)).astype(int)
train["oof"].value_counts()


# In[ ]:


np.savez_compressed('lgb_reg.npz',valid=oof_preds, test=y_preds)
     
f1_mean,f1_std = np.mean(cv_result),np.std(cv_result)
print(f"[CV] F1 Mean: {f1_mean}")
print(f"[CV] F1 Std: {f1_std}")


# make test predictions with optimized coefficients
sub_preds = np.round(np.clip(y_preds, 0, 10)).astype(int)
submission['open_channels'] = sub_preds
print(submission['open_channels'].value_counts()) 
submission.to_csv("submission.csv",index=False)

print(f1_score(y,np.round(np.clip(oof_preds, 0, 10)).astype(int),average='macro'))
print(cv_result)


# In[ ]:





# In[ ]:




