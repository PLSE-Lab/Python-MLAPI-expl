#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.covariance import OAS
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# In[ ]:


def MPEstimate(X, y, seed, n_clusters_per_class=3):
    emu, ep = [], []
    for yy in [0, 1]:
        XX = X[y==yy]
        ca = KMeans(n_clusters=n_clusters_per_class, random_state=seed)
        cluster = ca.fit_predict(XX)
        for i in range(n_clusters_per_class):
            C = XX[cluster==i]
            
            o = OAS().fit(C)
            mu = o.location_
            p  = o.precision_
            
            emu.append(mu)
            ep.append(p)
    return np.stack(emu), np.stack(ep)


# In[ ]:


def betterAucOrdering(data, preds, flips, eps = 1e-12, ascending=True):
    preds2 = preds.copy()
    rank = pd.Series(flips).rank(method='dense', ascending=ascending)
    
    for (r, g) in zip(rank, groups):
        idx = data[group]==g
        p = preds[idx].values
        p[p<0.5] += eps*r
        p[p>0.5] -= eps*r
        preds2[idx] = p
    return preds2


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

group  = 'wheezy-copper-turtle-magic'
target = 'target'
cols   = [c for c in train.columns if c not in ['id', target, group]]
groups = sorted(train[group].unique())


# In[ ]:


SEED = 2019

np.random.seed(SEED)
n_components = 6
n_clusters_per_class = n_components//2
score = []
flips = []

tr_preds = pd.Series(np.zeros(len(train)), index = train.index)
te_preds = pd.Series(np.zeros(len(test)), index = test.index)

for g in groups:
    tr_idx = train[group]==g
    te_idx = test[group]==g
        
    VT = VarianceThreshold(threshold=2)
    Xtr = VT.fit_transform(train.loc[tr_idx, cols])
    Xte = VT.transform(test.loc[te_idx, cols])
    Ytr = train.loc[tr_idx, target].values
    Xex = np.vstack((Xtr, Xte))
    

    ms, ps = MPEstimate(Xtr, Ytr, SEED)
    gm = GaussianMixture(n_components=n_components, init_params='random', covariance_type='full', 
                         tol=0.0001, max_iter=1000, n_init=1, random_state=SEED,
                         means_init=ms, precisions_init=ps)
    gm.fit(Xex)

    tr_preds[tr_idx] = gm.predict_proba(Xtr)[:, n_clusters_per_class:].sum(1)
    te_preds[te_idx] = gm.predict_proba(Xte)[:, n_clusters_per_class:].sum(1)
   
    diff = (Ytr!=tr_preds[tr_idx].round()).sum()
    flips.append(diff)

    
print('Train AUC - {}:'.format(SEED), roc_auc_score(train[target], tr_preds))


# In[ ]:


tr_preds2 = betterAucOrdering(train, tr_preds, flips, ascending=True)
print('Train Sorted AUC 1 - {}:'.format(SEED), roc_auc_score(train[target], tr_preds2))

tr_preds2 = betterAucOrdering(train, tr_preds, flips, ascending=False)
print('Train Sorted AUC 2 - {}:'.format(SEED), roc_auc_score(train[target], tr_preds2))

te_preds2 = betterAucOrdering(test, te_preds, flips, ascending=True) 


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub[target] = te_preds2
sub.to_csv('submission.csv', index=False)

