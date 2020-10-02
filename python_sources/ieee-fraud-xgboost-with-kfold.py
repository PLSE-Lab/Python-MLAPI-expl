#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
import itertools
import multiprocessing

from itertools import combinations
import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from pprint import pprint
from tqdm import tqdm

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


# In[ ]:


# better handle NAs (use timeseries to understand)

# integrate time splits in modelling

# create one global model and one local model


# # Preprocessing

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

def time_processing(df):
    startdate = datetime.datetime.strptime('2017-12-01', "%Y-%m-%d")
    df["Date"] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

    df['Weekdays'] = df['Date'].dt.dayofweek
    df['Hours'] = df['Date'].dt.hour
    df['Days'] = df['Date'].dt.day
    df = df.drop(['Date', 'TransactionDT'], axis=1)
    
    return(df)

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def mult_feat(df, feat1, feat2):
    feat_name = feat1+'x'+feat2
    df[feat_name] = -999
    booll = (df[feat1] != -999) & (df[feat2] != -999)
    df[feat_name][booll] = df[feat1][booll]*df[feat2][booll]
    
    return(df)


# In[ ]:


get_ipython().run_cell_magic('time', '', "##############################\n########## Get Data ##########\n##############################\nprint('\\nGet Data...')\nfiles = ['../input/test_identity.csv', '../input/test_transaction.csv',\n         '../input/train_identity.csv','../input/train_transaction.csv', \n        '../input/sample_submission.csv']\n\ndef load_data(file):\n    return pd.read_csv(file, index_col='TransactionID')\n\nwith multiprocessing.Pool() as pool:\n    test_identity, test_transaction, train_identity, train_transaction, sample_submission = pool.map(load_data, files)\n\n##############################\n######### Merge Data #########\n##############################\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ndel train_transaction, train_identity\ngc.collect()\n\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\ndel test_transaction, test_identity\ngc.collect()\n\n##############################\n######### Define Data ########\n##############################\nprint('\\nData Definition...')\n# Target\ny_train = train['isFraud'].copy()\nX_train = train.drop('isFraud', axis=1)\nX_test = test.copy()\n\ndel train, test\ngc.collect()\n\n# Features\ncat_features = ['ProductCD','addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']+ ['card' + str(i) for i in range(1,7)] + ['M' + str(i) for i in range(1,10)] + ['DeviceType','DeviceInfo'] + ['id_' + str(i) for i in range(12,39)] \nnum_features = [i for i in X_train.columns if i not in cat_features]        \n\n##############################\n####### Missing Values ######\n##############################\nprint('\\nMissing Values...')\nX_train[cat_features] = X_train[cat_features].fillna('missing')\nX_test[cat_features] = X_test[cat_features].fillna('missing')\n\nX_train[num_features] = X_train[num_features].fillna(-999)\nX_test[num_features] = X_test[num_features].fillna(-999)\n\n##############################\n####### Time Features ########\n##############################\nprint('\\nTime Features...')\nX_train, X_test = time_processing(X_train), time_processing(X_test)\n\ncat_features += ['Weekdays', 'Hours', 'Days']\n\n##############################\n###### Features Encoding #####\n##############################\n\ncat_features_dummy = ['ProductCD', 'DeviceType', 'card4', 'card6', 'Hours']+['M' + str(i) for i in range(1,10)]\n\nif len(cat_features_dummy) != 0:\n    print('\\nOne Hot Encoding...')\n    X_train = pd.get_dummies(X_train, prefix = cat_features_dummy, columns = cat_features_dummy, sparse=True)\n    X_test = pd.get_dummies(X_test, prefix = cat_features_dummy, columns = cat_features_dummy, sparse=True)\n\nprint('\\nLabel Encoding...')\nfor f in cat_features:\n    if f not in cat_features_dummy:\n        lbl = preprocessing.LabelEncoder()\n        lbl.fit(list(X_train[f].values) + list(X_test[f].values))\n        X_train[f] = lbl.transform(list(X_train[f].values))\n        X_test[f] = lbl.transform(list(X_test[f].values))  \n  \ncols = intersection(X_train.columns, X_test.columns)\nX_train, X_test = X_train[cols], X_test[cols]\n\n##############################\n######## Feature Eng #########\n##############################\nprint('\\nFeature Engineering...')\n\npd.set_option('mode.chained_assignment', None)\nfor feat1, feat2 in combinations(['V201', 'V258', 'V257', 'V244', 'V189', 'V246'], 2):\n    X_train, X_test = mult_feat(X_train.copy(), feat1, feat2), mult_feat(X_test.copy(), feat1, feat2)\n\n##############################\n######## Reduce Memory #######\n##############################\nprint('\\nReducing Memory... \\n')\nX_train = reduce_mem_usage(X_train)\nX_test = reduce_mem_usage(X_test)")


# ## Modelling

# In[ ]:


def augment(x,y,feat,t=2):
    xs = []
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    xs = np.vstack(xs)
    ys = np.ones(xs.shape[0])
    x = np.vstack([x,xs])
    x = pd.DataFrame(x, columns = feat)
    y = np.concatenate([y,ys])
    return x,y


# In[ ]:


class XGBGridSearch:

    def __init__(self, param_grid, cv=3, verbose=0, shuffle=False, random_state=2019, augment = False):
        self.param_grid = param_grid
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.shuffle = shuffle
        self.augment = augment
        
        self.average_scores = []
        self.scores = []
        self.feature_importance_df = pd.DataFrame()
    
    def fit(self, X, y):
        self._expand_params()
        self._split_data(X, y)
            
        for params in tqdm(self.param_list, disable=not self.verbose):
            avg_score, score = self._run_cv(X, y, params)
            self.average_scores.append(avg_score)
            self.scores.append(score)
        
        self._compute_best()
        
    def _run_cv(self, X, y, params):
        scores = []
        
        for fold_, (train_idx, val_idx) in enumerate(self.splits):
            clf = xgb.XGBClassifier(**params)

            X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if self.augment:
                X_train, y_train = augment(X_train.values, y_train.values, X_train.columns)
                
            clf.fit(X_train, y_train)
            
            self.avg_feat_importance(features = [c for c in X_train.columns], clf = clf, fold_ = fold_)
            
            y_val_pred = clf.predict_proba(X_val)[:, 1]
            
            score = roc_auc_score(y_val, y_val_pred)
            scores.append(score)
            
            gc.collect()
        
        avg_score = sum(scores) / len(scores)
        return avg_score, scores
            
    def _split_data(self, X, y):
        kf = KFold(n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state)
        self.splits = list(kf.split(X, y))
            
    def _compute_best(self):
        idx_best = np.argmax(self.average_scores)
        self.best_score_ = self.average_scores[idx_best]
        self.best_params_ = self.param_list[idx_best]

    def _expand_params(self):
        keys, values = zip(*self.param_grid.items())
        self.param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
    def avg_feat_importance(self, features, clf, fold_):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        self.feature_importance_df = pd.concat([self.feature_importance_df, fold_importance_df], axis=0)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nparam_grid = {\n    \'n_estimators\': [500],\n    \'missing\': [-999],\n    \'random_state\': [2019],\n    \'n_jobs\': [1],\n    \'tree_method\': [\'gpu_hist\'],\n    \'max_depth\': [9],\n    \'learning_rate\': [0.05],\n    \'subsample\': [0.9],\n    \'colsample_bytree\': [0.9],\n    \'reg_alpha\': [0],\n    \'reg_lambda\': [1]\n}\n\ngrid = XGBGridSearch(param_grid, cv=4, verbose=1, augment = True)\ngrid.fit(X_train, y_train)\n\nprint("Best Score:", grid.best_score_)\nprint("Best Params:", grid.best_params_)\n\nclf = xgb.XGBClassifier(**grid.best_params_)\nclf.fit(X_train, y_train)\n\nsample_submission[\'isFraud\'] = clf.predict_proba(X_test)[:,1]\nsample_submission.to_csv(\'simple_no_time_xgboost.csv\')')


# In[ ]:


# inspired from link below
# https://www.kaggle.com/jesucristo/santander-magic-lgb-0-901

feature_importance_df = grid.feature_importance_df

cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:100].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('XGBoost Features (averaged over folds)')
plt.tight_layout()

