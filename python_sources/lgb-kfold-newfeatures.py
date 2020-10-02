#!/usr/bin/env python
# coding: utf-8

#  From notebooks of
#  https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s  
#  https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm
# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda

# In[ ]:


print('loading libs...')
import warnings
warnings.filterwarnings("ignore")
import os
import gc
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import time
print('done')


# In[ ]:


get_ipython().run_cell_magic('time', '', "print('loading data...')\ntrain = pd.read_pickle('../input/ieee-fe-with-some-eda/train_df.pkl')\ntest = pd.read_pickle('../input/ieee-fe-with-some-eda/test_df.pkl')\nremove_features = pd.read_pickle('../input/ieee-fe-with-some-eda/remove_features.pkl')\nsample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')\nprint('done')")


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "print('dropping target...')\ny_train = train['isFraud'].copy()\nX_train = train.drop('isFraud', axis=1)\nX_test = test.copy()\ntrain_cols = list(train.columns)\ndel train, test\ngc.collect()\nprint('selecting features...')\nremove_features = list(remove_features['features_to_remove'].values)\nfeatures_columns = [col for col in train_cols if col not in remove_features]\nX_train = X_train[features_columns]\nX_test=X_test[features_columns]\nprint('Done')")


# In[ ]:


X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)


# In[ ]:


params = {
          'objective':'binary',
          'boosting_type':'gbdt',
          'metric':'auc',
          'n_jobs':-1,
          'max_depth':-1,
          'tree_learner':'serial',
          'min_data_in_leaf':30,
          'n_estimators':1800,
          'max_bin':255,
          'verbose':-1,
          'seed': 1229,
          'learning_rate': 0.01,
          'early_stopping_rounds':200,
          'colsample_bytree': 0.5,          
          'num_leaves': 256, 
          'reg_alpha': 0.35, 
         }


# In[ ]:


get_ipython().run_cell_magic('time', '', 'NFOLDS = 6\nfolds = KFold(n_splits=NFOLDS)\ncolumns = X_train.columns\nsplits = folds.split(X_train, y_train)\ny_preds = np.zeros(X_test.shape[0])\ny_oof = np.zeros(X_train.shape[0])\nscore = 0\n  \nfor fold_n, (train_index, valid_index) in enumerate(splits):\n    X_tr, X_val = X_train[columns].iloc[train_index], X_train[columns].iloc[valid_index]\n    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]    \n    dtrain = lgb.Dataset(X_tr, label=y_tr)\n    dvalid = lgb.Dataset(X_val, label=y_val)\n    clf = lgb.train(params, dtrain,  valid_sets = [dtrain, dvalid], verbose_eval=500)        \n    y_pred_valid = clf.predict(X_val)\n    y_oof[valid_index] = y_pred_valid\n    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_val, y_pred_valid)}")   \n    score += roc_auc_score(y_val, y_pred_valid) / NFOLDS\n    del X_tr, X_val, y_tr, y_val\n    gc.collect() \n    y_preds += clf.predict(X_test) / NFOLDS       \nprint(f"\\nMean AUC = {score}")\nprint(f"Out of folds AUC = {roc_auc_score(y_train, y_oof)}")\n\nprint(\'submission...\')\nsample_submission[\'isFraud\'] = y_preds\nsample_submission.to_csv("submission_lgb.csv", index=False)')


# In[ ]:




