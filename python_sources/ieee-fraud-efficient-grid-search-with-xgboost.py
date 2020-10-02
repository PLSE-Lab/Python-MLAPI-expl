#!/usr/bin/env python
# coding: utf-8

# # About this Kernel
# 
# The goal with this kernel is to share a simple, reusable, and sklearn-like class for you to use grid search on XGBoost, **without running out of RAM**. The difference between `XGBGridSearch` and scikit-learn's `GridSearchCV` is that my class does not make multiple copies of the data, and is more aggressive in terms of memory management (it calls `gc.collect()` after every run). **Otherwise, it should feel just like `GridSearchCV`.**
# 
# ## Credits
# 
# I'd be happy if you would like to fork, modify and share this kernel. If you do so, or if you are copying a certain function or class from here, please do cite or link to this kernel directly. This would be highly appreciated, since I spent a lot of time working on this.
# 
# ## References
# * [KFold CV + RAM Optimization](https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb)
# * [Using XGBoost with GPU](https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s)
# * [Freeing up memory from GPU](https://github.com/dmlc/xgboost/issues/3045)

# In[ ]:


import os
import gc
import itertools

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from pprint import pprint
from tqdm import tqdm


# # Basic Preprocessing

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')\n\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\n\nsample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\n\nprint(train.shape)\nprint(test.shape)\n\ny_train = train['isFraud'].copy()\ndel train_transaction, train_identity, test_transaction, test_identity\n\n# Drop target, fill in NaNs\nX_train = train.drop(['isFraud', 'TransactionDT'], axis=1)\nX_test = test.drop(['TransactionDT'], axis=1)\ndel train, test\n\nX_train = X_train.fillna(-999)\nX_test = X_test.fillna(-999)\n\n# Label Encoding\nfor f in X_train.columns:\n    if X_train[f].dtype=='object' or X_test[f].dtype=='object': \n        lbl = preprocessing.LabelEncoder()\n        lbl.fit(list(X_train[f].values) + list(X_test[f].values))\n        X_train[f] = lbl.transform(list(X_train[f].values))\n        X_test[f] = lbl.transform(list(X_test[f].values))   ")


# # RAM Optimization

# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train = reduce_mem_usage(X_train)\nX_test = reduce_mem_usage(X_test)')


# # XGBGridSearch Class

# In[ ]:


class XGBGridSearch:
    """
    Source:
    https://www.kaggle.com/xhlulu/ieee-fraud-efficient-grid-search-with-xgboost
    """
    def __init__(self, param_grid, cv=3, verbose=0, 
                 shuffle=False, random_state=2019):
        self.param_grid = param_grid
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.shuffle = shuffle
        
        self.average_scores = []
        self.scores = []
    
    def fit(self, X, y):
        self._expand_params()
        self._split_data(X, y)
            
        for params in tqdm(self.param_list, disable=not self.verbose):
            avg_score, score = self._run_cv(X, y, params)
            self.average_scores.append(avg_score)
            self.scores.append(score)
        
        self._compute_best()

    def _run_cv(self, X, y, params):
        """
        Perform KFold CV on a single set of parameters
        """
        scores = []
        
        for train_idx, val_idx in self.splits:
            clf = xgb.XGBClassifier(**params)

            X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            clf.fit(X_train, y_train)
            
            y_val_pred = clf.predict_proba(X_val)[:, 1]
            
            score = roc_auc_score(y_val, y_val_pred)
            scores.append(score)
            
            gc.collect()
        
        avg_score = sum(scores) / len(scores)
        return avg_score, scores
            
    def _split_data(self, X, y):
        kf = KFold(n_splits=self.cv, 
                   shuffle=self.shuffle, 
                   random_state=self.random_state)
        self.splits = list(kf.split(X, y))
            
    def _compute_best(self):
        """
        Compute best params and its corresponding score
        """
        idx_best = np.argmax(self.average_scores)
        self.best_score_ = self.average_scores[idx_best]
        self.best_params_ = self.param_list[idx_best]

    def _expand_params(self):
        """
        This method expands a dictionary of lists into
        a list of dictionaries (each dictionary is a single
        valid params that can be input to XGBoost)
        """
        keys, values = zip(*self.param_grid.items())
        self.param_list = [
            dict(zip(keys, v)) 
            for v in itertools.product(*values)
        ]


# # Training
# 
# You will find below the `param_grid` with the default input parameters to XGBoost. Please add more element to the desired list in order to run the model.

# In[ ]:


param_grid = {
    'n_estimators': [500],
    'missing': [-999],
    'random_state': [2019],
    'n_jobs': [1],
    'tree_method': ['gpu_hist'],
    'max_depth': [9],
    'learning_rate': [0.048, 0.05],
    'subsample': [0.85, 0.9],
    'colsample_bytree': [0.85, 0.9],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 0.9]
}

grid = XGBGridSearch(param_grid, cv=4, verbose=1)
grid.fit(X_train, y_train)

print("Best Score:", grid.best_score_)
print("Best Params:", grid.best_params_)


# ## Refit and Submit

# In[ ]:


clf = xgb.XGBClassifier(**grid.best_params_)
clf.fit(X_train, y_train)

sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_xgboost.csv')

