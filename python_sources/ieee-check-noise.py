#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

import math
warnings.filterwarnings('ignore')


# In[ ]:


########################### Helpers
#################################################################################
## -------------------
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
## ------------------- 


# In[ ]:


########################### Vars
#################################################################################
SEED = 42
seed_everything(SEED)
LOCAL_TEST = True
TARGET = 'isFraud'


# In[ ]:


########################### DATA LOAD
#################################################################################
print('Load Data')
train_df = pd.read_pickle('../input/ieee-data-minification/train_transaction.pkl')

if LOCAL_TEST:
    test_df = train_df.iloc[-100000:,].reset_index(drop=True)
    train_df = train_df.iloc[:400000,].reset_index(drop=True)
    
    train_identity = pd.read_pickle('../input/ieee-data-minification/train_identity.pkl')
    test_identity  = train_identity[train_identity['TransactionID'].isin(test_df['TransactionID'])].reset_index(drop=True)
    train_identity = train_identity[train_identity['TransactionID'].isin(train_df['TransactionID'])].reset_index(drop=True)
else:
    test_df = pd.read_pickle('../input/ieee-data-minification/test_transaction.pkl')
    test_identity = pd.read_pickle('../input/ieee-data-minification/test_identity.pkl')


# In[ ]:


########################### Model params
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':1,
                    'n_estimators':800,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                } 


# In[ ]:


########################### Model
import lightgbm as lgb

def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  

    tt_df = tt_df[['TransactionID',target]]    
    predictions = np.zeros(len(tt_df))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold:',fold_)
        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
            
        print(len(tr_x),len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        if LOCAL_TEST:
            vl_data = lgb.Dataset(P, label=P_y) 
        else:
            vl_data = lgb.Dataset(vl_x, label=vl_y)  

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets = [tr_data, vl_data],
            verbose_eval = 200,
        )   
        
        pp_p = estimator.predict(P)
        predictions += pp_p/NFOLDS

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])
            print(feature_imp)
        
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()
        
    tt_df['prediction']  = predictions
    
    return tt_df
## -------------------


# In[ ]:


########################### Ground Base line
features_columns = ['M1','M2','M3','M5','M6','M7','M8','M9']
test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)


# In[ ]:


########################### With Transaction Amount
features_columns = ['M1','M2','M3','M5','M6','M7','M8','M9', 'TransactionAmt']
test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)


# In[ ]:


# As you can see TransactionAmt is very important feature
#  2710  TransactionAmt
# Lets change it to random noise

mu, sigma = 0,1
noise = np.random.normal(mu, sigma, len(train_df))
train_df['TransactionAmt_noise'] = noise

noise = np.random.normal(mu, sigma, len(test_df))
test_df['TransactionAmt_noise'] = noise

features_columns = ['M1','M2','M3','M5','M6','M7','M8','M9', 'TransactionAmt_noise']
test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)


# In[ ]:


# 2804  TransactionAmt_noise 
# Almost same as original Transaction Amount

