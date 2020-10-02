#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from tqdm import tqdm_notebook

import os

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# In[ ]:


from scipy.special import logit, expit
eps = 1e-10

def normalise(pred):
    m = np.median(pred)
    norm_pred = expit(logit(pred * (1 - 2 * eps) + eps) - logit(m))
    return norm_pred


# In[ ]:


from sklearn.mixture import GaussianMixture
from sklearn import covariance
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from scipy.stats import rankdata
import time


def re_label(arr, n_min = 5):
    counts = np.bincount(arr)
    while np.min(counts) < n_min:
        min_lab = np.argmin(counts)
        max_lab = np.argmax(counts)
        arr[arr == min_lab] = max_lab
        arr[arr > min_lab] = arr[arr > min_lab] - 1
        counts = np.bincount(arr)
    return arr


def fit_model(train_X, train_y, test_X, data, n_clusters=2):
    
    oof = np.zeros(len(train_X))
    preds = np.zeros(len(test_X))

    skf = StratifiedKFold(n_splits=11, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(train_X, train_y):
        
        target = train_y[train_index]
        y_train = target.copy()
        
        neg_pred = GaussianMixture(n_clusters, n_init=1, max_iter=10000).fit_predict(train_X[train_index][target==0])
        neg_pred = re_label(neg_pred)
        n_neg_clusters = np.max(neg_pred) + 1
        y_train[target == 0] = neg_pred

        pos_pred = GaussianMixture(n_clusters, n_init=1, max_iter=10000).fit_predict(train_X[train_index][target==1])
        pos_pred = re_label(pos_pred)
        n_pos_clusters = np.max(pos_pred) + 1
        y_train[target == 1] = pos_pred + n_neg_clusters
        
        covs = [covariance.OAS().fit(train_X[train_index][y_train==j]) for j in np.unique(y_train)]
        clf1 = GaussianMixture(n_neg_clusters+n_pos_clusters, init_params='kmeans', max_iter=100000,
                              means_init=[cov.location_ for cov in covs],
                              precisions_init=[cov.precision_ for cov in covs]).fit(np.vstack([train_X, test_X]))
        
        oof[test_index] = clf1.predict_proba(train_X[test_index,:])[:,n_neg_clusters:].sum(axis=1)
        #oof[test_index] = rankdata(oof[test_index]) / len(test_index)
        oof[test_index] = normalise(oof[test_index])
        #preds += rankdata(clf1.predict_proba(test_X)[:,n_neg_clusters:].sum(axis=1)) / len(test_X) / skf.n_splits
        preds += normalise(clf1.predict_proba(test_X)[:,n_neg_clusters:].sum(axis=1)) 
    
    return oof, preds


def fiti(train, test):
    idx1 = train.index; idx2 = test.index
    train2 = train.reset_index(drop=True,inplace=False); test2 = test.reset_index(drop=True,inplace=False)

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=2).fit_transform(data[cols]))
    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]
    
    return fit_model(train3, train2['target'].values, test3, data2, 2)


bag_size = 20

oofs = np.zeros((len(train), bag_size))
preds = np.zeros((len(test), bag_size))

start_time = time.time()

buckets = range(512 * bag_size)

result = Parallel(n_jobs=4, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")    (delayed(fiti)(train[train['wheezy-copper-turtle-magic']==i%512], test[test['wheezy-copper-turtle-magic']==i%512]) for i in tqdm_notebook(buckets))

for i, res in enumerate(result):
    
    oofs[train['wheezy-copper-turtle-magic']==i%512, i//512] += res[0] 
    preds[test['wheezy-copper-turtle-magic']==i%512, i//512] += res[1] 

oof = np.mean(oofs, axis=1)
pred = np.mean(preds, axis=1)
    
#print()
#print(f"Elapsed time:{time.time() - start_time:.5}")

print()
print(f'{roc_auc_score(train["target"], oof):.5}')


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv', index=False)


# In[ ]:


from numba import jit

@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


# In[ ]:


print()
print(f'{roc_auc_score(train["target"], oof):.5}')
print(f'{fast_auc(train["target"], oof):.5}')


# In[ ]:


oofs

