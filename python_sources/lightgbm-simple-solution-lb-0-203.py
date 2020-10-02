#!/usr/bin/env python
# coding: utf-8

# >v0.1 This code implements a simple feature extraction and train using Lightgbm.
# 
# Feature extraction is very simple and can be improved.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import librosa
import matplotlib.pyplot as plt
import gc

from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed
import lightgbm as lgb
from scipy import stats

from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

tqdm.pandas()


# In[ ]:


def split_and_label(rows_labels):
    
    row_labels_list = []
    for row in rows_labels:
        row_labels = row.split(',')
        labels_array = np.zeros((80))
        
        for label in row_labels:
            index = label_mapping[label]
            labels_array[index] = 1
        
        row_labels_list.append(labels_array)
    
    return row_labels_list


# In[ ]:


train_curated = pd.read_csv('../input/train_curated.csv')
train_noisy = pd.read_csv('../input/train_noisy.csv')
train_noisy = train_noisy[['fname','labels']]
test = pd.read_csv('../input/sample_submission.csv')
print(train_curated.shape, train_noisy.shape, test.shape)


# In[ ]:


label_columns = list( test.columns[1:] )
label_mapping = dict((label, index) for index, label in enumerate(label_columns))
label_mapping


# In[ ]:


train_curated_labels = split_and_label(train_curated['labels'])
train_noisy_labels   = split_and_label(train_noisy  ['labels'])
len(train_curated_labels), len(train_noisy_labels)


# In[ ]:


for f in label_columns:
    train_curated[f] = 0.0
    train_noisy[f] = 0.0

train_curated[label_columns] = train_curated_labels
train_noisy[label_columns]   = train_noisy_labels

train_curated['num_labels'] = train_curated[label_columns].sum(axis=1)
train_noisy['num_labels']   = train_noisy[label_columns].sum(axis=1)

train_curated['path'] = '../input/train_curated/'+train_curated['fname']
train_noisy  ['path'] = '../input/train_noisy/'+train_noisy['fname']

train_curated.head()


# In[ ]:


train = pd.concat([train_curated, train_noisy],axis=0)

del train_curated, train_noisy
gc.collect()

train.shape


# In[ ]:


def create_features( pathname ):

    var, sr = librosa.load( pathname, sr=44100)
    # trim silence
    if 0 < len(var): # workaround: 0 length causes error
        var, _ = librosa.effects.trim(var)
    xc = pd.Series(var)
    
    X = []
    X.append( xc.mean() )
    X.append( xc.median() )
    X.append( xc.std() )
    X.append( xc.max() )
    X.append( xc.min() )
    X.append( xc.skew() )
    X.append( xc.mad() )
    X.append( xc.kurtosis() )
    
    X.append( np.mean(np.diff(xc)) )
    X.append( np.mean(np.nonzero((np.diff(xc) / xc[:-1]))[0]) )
    X.append( np.abs(xc).max() )
    X.append( np.abs(xc).min() )
    
    X.append( xc[:4410].std() )
    X.append( xc[-4410:].std() )
    X.append( xc[:44100].std() )
    X.append( xc[-44100:].std() )
    
    X.append( xc[:4410].mean() )
    X.append( xc[-4410:].mean() )
    X.append( xc[:44100].mean() )
    X.append( xc[-44100:].mean() )
    
    X.append( xc[:4410].min() )
    X.append( xc[-4410:].min() )
    X.append( xc[:44100].min() )
    X.append( xc[-44100:].min() )
    
    X.append( xc[:4410].max() )
    X.append( xc[-4410:].max() )
    X.append( xc[:44100].max() )
    X.append( xc[-44100:].max() )
    
    X.append( xc[:4410].skew() )
    X.append( xc[-4410:].skew() )
    X.append( xc[:44100].skew() )
    X.append( xc[-44100:].skew() )
    
    X.append( xc.max() / np.abs(xc.min()) )
    X.append( xc.max() - np.abs(xc.min()) )
    X.append( xc.sum() )
    
    X.append( np.mean(np.nonzero((np.diff(xc[:4410]) / xc[:4410][:-1]))[0]) )
    X.append( np.mean(np.nonzero((np.diff(xc[-4410:]) / xc[-4410:][:-1]))[0]) )
    X.append( np.mean(np.nonzero((np.diff(xc[:44100]) / xc[:44100][:-1]))[0]) )
    X.append( np.mean(np.nonzero((np.diff(xc[-44100:]) / xc[-44100:][:-1]))[0]) )
    
    X.append( np.quantile(xc, 0.95) )
    X.append( np.quantile(xc, 0.99) )
    X.append( np.quantile(xc, 0.10) )
    X.append( np.quantile(xc, 0.05) )
    
    X.append( np.abs(xc).mean() )
    X.append( np.abs(xc).std() )
             
    return np.array( X )


# In[ ]:



X = Parallel(n_jobs= 4)(delayed(create_features)(fn) for fn in tqdm(train['path'].values) )
X = np.array( X )
X.shape


# In[ ]:


Xtest = Parallel(n_jobs= 4)(delayed(create_features)( '../input/test/'+fn) for fn in tqdm(test['fname'].values) )
Xtest = np.array( Xtest )
Xtest.shape


# In[ ]:



n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=69)

params = {'num_leaves': 15,
         'min_data_in_leaf': 200, 
         'objective':'binary',
         "metric": 'auc',
         'max_depth': -1,
         'learning_rate': 0.05,
         "boosting": "gbdt",
         "bagging_fraction": 0.85,
         "bagging_freq": 1,
         "feature_fraction": 0.20,
         "bagging_seed": 42,
         "verbosity": -1,
         "nthread": -1,
         "random_state": 69}

PREDTRAIN = np.zeros( (X.shape[0],80) )
PREDTEST  = np.zeros( (Xtest.shape[0],80) )
for f in range(len(label_columns)):
    y = train[ label_columns[f] ].values
    oof      = np.zeros( X.shape[0] )
    oof_test = np.zeros( Xtest.shape[0] )
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X,y)):
        model = lgb.LGBMClassifier(**params, n_estimators = 20000)
        model.fit(X[trn_idx,:], 
                  y[trn_idx], 
                  eval_set=[(X[val_idx,:], y[val_idx])], 
                  eval_metric='auc',
                  verbose=0, 
                  early_stopping_rounds=25)
        oof[val_idx] = model.predict_proba(X[val_idx,:], num_iteration=model.best_iteration_)[:,1]
        oof_test += model.predict_proba(Xtest          , num_iteration=model.best_iteration_)[:,1]/5.0

    PREDTRAIN[:,f] = oof    
    PREDTEST [:,f] = oof_test
    
    print( f, str(roc_auc_score( y, oof ))[:6], label_columns[f] )


# In[ ]:


from sklearn.metrics import roc_auc_score
def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = label_ranking_average_precision_score(
        truth[nonzero_weight_sample_indices, :] > 0, 
        scores[nonzero_weight_sample_indices, :], 
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap

print( 'lwlrap cv:', calculate_overall_lwlrap_sklearn( train[label_columns].values, PREDTRAIN ) )


# In[ ]:


test[label_columns] = PREDTEST
test.to_csv('submission.csv', index=False)
test.head()


# In[ ]:




