#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[3]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[4]:


x_cols = ['var_%d'%i for i in range(200)]


# In[5]:


X_train = train_df[x_cols].values
y_train = train_df['target'].values
X_test = test_df[x_cols].values


# In[8]:


from sklearn.naive_bayes import MultinomialNB, GaussianNB,ComplementNB
from sklearn.preprocessing import Normalizer, StandardScaler

def model_gauss(trn_x, trn_y, val_x, val_y, text_x):
    clf = GaussianNB()
    clf.fit(trn_x, trn_y)
    val_pred = clf.predict_proba(val_x)[:, 1]
    test_fold_pred = clf.predict_proba(text_x)[:, 1]
    return val_pred, test_fold_pred


# In[ ]:


from sklearn.model_selection import StratifiedKFold
import time

INNER_FOLDS = 3
OUTER_FOLDS = 5

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)

test_preds_sum = np.zeros((X_test.shape[0], 1))
fold_val_roc = []
for fold_, (trn_, val_) in enumerate(folds.split(y_train, y_train)):
    print('Fold:', fold_)
    trn_x, trn_y = X_train[trn_, :], y_train[trn_]
    val_x, val_y = X_train[val_, :], y_train[val_]
    
    trn_y_zeros_mask = (trn_y == 0)
    
    trn_x_ones = trn_x[~trn_y_zeros_mask, : ]
    trn_y_ones = trn_y[~trn_y_zeros_mask]
    
    trn_x_zeros = trn_x[trn_y_zeros_mask, : ]
    trn_y_zeros = trn_y[trn_y_zeros_mask]

    fold_val = np.zeros((len(val_y), 1))
    for fold_2, (_, zeros_sample) in enumerate(folds_inner.split(trn_y_zeros, trn_y_zeros)):
        print("Inner Fold", fold_2 + 1, 'of', INNER_FOLDS)
        trn_x_2 = np.vstack([trn_x_ones, trn_x_zeros[zeros_sample, :]])
        trn_y_2 = np.hstack([trn_y_ones, trn_y_zeros[zeros_sample]])
        
        print('Training:')
        s = time.time()
        val_preds, test_preds = model_gauss(trn_x_2, trn_y_2, val_x, val_y, X_test)
        print('Training Time', time.time() - s)
        
        print('AUC: ', roc_auc_score(val_y, val_preds))
        
        fold_val += val_preds.reshape((-1, 1))
        test_preds_sum += test_preds.reshape((-1, 1))
    val_roc = roc_auc_score(val_y, fold_val.flatten()/INNER_FOLDS)
    fold_val_roc.append( val_roc )
    print('Fold Val: ', fold_, 'AUC:', val_roc)
print('Mean Val ROC', np.array(fold_val_roc).mean())


# In[ ]:


sub_df = pd.DataFrame({'ID_code': test_df['ID_code'].values})
sub_df['target'] = (test_preds_sum.flatten()/(OUTER_FOLDS * INNER_FOLDS))
sub_df.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




