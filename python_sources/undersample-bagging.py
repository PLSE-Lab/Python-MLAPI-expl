#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import lightgbm as lgb


# # Model Definition

# In[ ]:


#UnderSampleLGBM
def UnderSampleLGBM(X_train, y_train, X_valid, X_test, n_estimators, random_state = 22):
    clf = LGBMClassifier(
                        n_estimators= 15000,
                        bagging_freq= 5,          
                        bagging_fraction= 0.38,   
                        boost_from_average='false',   
                        boost= 'gbdt',             
                        feature_fraction= 0.04,     
                        learning_rate= 0.0085,
                        max_depth= -1,             
                        metric='auc',                
                        min_data_in_leaf= 80,     
                        min_sum_hessian_in_leaf= 10.0,
                        num_leaves= 13,            
                        num_threads= 8,              
                        tree_learner= 'serial',   
                        objective= 'binary',
                        reg_alpha= 0.1302650970728192, 
                        reg_lambda= 0.3603427518866501,
                        verbosity= 1,
                        n_jobs = -1)
    clfs = [deepcopy(clf) for i in range(n_estimators)]
    np.random.seed(seed = 43)
    pred_val = np.zeros(len(X_valid))
    pred_test = np.zeros(len(X_test))
    
    # Train cloned base models
    for i, __clf in enumerate(clfs):   
        rs = int(np.random.rand() * 255) + i
        sampler = RandomUnderSampler(random_state=rs, replacement=True)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
        for train_index, test_index in sss.split(X_resampled, y_resampled):
            X_train2, X_valid2 = X_resampled[train_index], X_resampled[test_index]
            y_train2, y_valid2 = y_resampled[train_index], y_resampled[test_index]         
        print('training...', i)
        __clf.fit(X_train2, y_train2, eval_set = (X_valid2, y_valid2), verbose = False, early_stopping_rounds = 1000)
        print('fitting')
        pred_val += __clf.predict_proba(X_valid,num_iteration = __clf.best_iteration_)[:,1]/ n_estimators
        pred_test += __clf.predict_proba(X_test,num_iteration = __clf.best_iteration_)[:,1]/ n_estimators
        #pred_val = np.column_stack([__clf.predict_proba(X_valid,num_iteration = __clf.best_iteration_)[:,1] for __clf in clfs])
    return pred_val, pred_test


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


X = train_df.drop(['target', 'ID_code'], axis = 1)
y = train_df.target
X_test = test_df.drop(['ID_code'], axis = 1)


# # K-fold CV

# In[ ]:


#parameters
num_fold = 12
n_estimators = 20


# In[ ]:


#training
folds = StratifiedKFold(n_splits=num_fold, shuffle=False, random_state=42)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    pred_val, pred_test = UnderSampleLGBM(X.iloc[trn_idx], y.iloc[trn_idx], X.iloc[val_idx], X_test, n_estimators)
    predictions += pred_test / folds.n_splits
    oof[val_idx] = pred_val
    print('AUC of FOLD',fold_,' : ',roc_auc_score(y.iloc[val_idx], pred_val))
print('AUC of CV : ',roc_auc_score(y, oof) )


# In[ ]:


test_df['target'] = predictions
test_df[['ID_code', 'target']].to_csv('./sub_3.csv', index = False)

