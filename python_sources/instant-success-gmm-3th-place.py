#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from pathlib import Path
import csv
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm
import pdb
import lightgbm as lgb
import xgboost as xgb
import random
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
import seaborn as sn
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.covariance import GraphicalLasso
from sklearn.feature_selection import VarianceThreshold
from scipy import linalg

PATH_BASE = Path('../input')
PATH_WORKING = Path('../working')


# In[ ]:


train = pd.read_csv(PATH_BASE/'train.csv')
test = pd.read_csv(PATH_BASE/'test.csv')


# In[ ]:


def get_mean_cov(x,y):
    model = GraphicalLasso(max_iter=200)
    ones = (y==1).astype(bool)
    x2 = x[ones]
    model.fit(x2)
    p1 = model.precision_
    m1 = model.location_
    
    onesb = (y==0).astype(bool)
    x2b = x[onesb]
    model.fit(x2b)
    p2 = model.precision_
    m2 = model.location_
    
    ms = np.stack([m1,m2])
    ps = np.stack([p1,p2])
    return ms,ps


# In[ ]:


def projectMeans(means):
    means[means>0]=1
    means[means<=0]=-1
    return means

def _compute_precision_cholesky(covariances, covariance_type):
    estimate_precision_error_message = ("Hell no")
    
    if covariance_type in 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
    
    return precisions_chol

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances

def _estimate_gaussian_parameters2(X, resp, reg_covar, covariance_type):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    means = projectMeans(means)

    covariances = {"full": _estimate_gaussian_covariances_full}[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances

class GaussianMixture2(GaussianMixture):
    def _m_step(self, X, log_resp):
        resp = np.exp(log_resp)
        sums = resp.sum(0)
        if sums.max() - sums.min() > 2:
            for i in range(3):
                resp = len(X) * resp / resp.sum(0) / len(sums)
                resp = resp/resp.sum(1)[:,None]
        
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters2(X, resp, self.reg_covar,
                                          self.covariance_type))
        
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)

        
random.seed(1234)
np.random.seed(1234)
os.environ['PYTHONHASHSEED'] = str(1234)


# In[ ]:


cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
oof = np.zeros(len(train))
preds = np.zeros(len(test))

N_RAND_INIT = 3
N_CLUST_OPT = 3
N_TEST = 1

all_acc = np.zeros((512, N_CLUST_OPT, N_RAND_INIT))
all_roc = np.zeros((512, N_CLUST_OPT, N_RAND_INIT))
cluster_cnt = np.zeros((512, N_CLUST_OPT, N_RAND_INIT))
cluster_div = np.zeros((512, N_CLUST_OPT, N_RAND_INIT))

j_selection = np.zeros(N_CLUST_OPT)

for i in tqdm(range(512)):
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    
    test_index = range(len(train3))

    yf = train2['target']
    ms, ps = get_mean_cov(train3,yf)
    
    cc_list = []
    nc_list = 2*(np.array(range(N_CLUST_OPT)) + 2)
    
    for j in range(N_CLUST_OPT):
        cc_list.append(['cluster_' + str(i) for i in range(nc_list[j])])
    
    gm_list = []
    acc = np.zeros((N_CLUST_OPT, N_RAND_INIT))
    res_list = []
    ctc_list = []
    
    for j in range(N_CLUST_OPT):
        
        gm_list.append([])
        res_list.append([])
        ctc_list.append([])
        
        nc = nc_list[j]
        cl = int(0.5*nc)
        
        for k in range(N_RAND_INIT):
            ps_list = np.concatenate([ps]*cl, axis=0)

            th_step = 100/(cl+1)
            th_p = np.arange(th_step,99,th_step) + 0.5*(np.random.rand(cl) - 0.5)*th_step
            th = np.percentile(ms,th_p)

            ms_list = []
            for t in range(cl):
                ms_new = ms.copy()
                ms_new[ms>=th[t]]=1
                ms_new[ms<th[t]]=-1
                ms_list.append(ms_new)
            ms_list = np.concatenate(ms_list, axis=0)
            
            perm = np.random.permutation(nc)
            ps_list = ps_list[perm]
            ms_list = ms_list[perm]
            
            gm = GaussianMixture2(n_components=nc, init_params='random', covariance_type='full', tol=0.0001,reg_covar=0.1,
                                  max_iter=5000, n_init=1, means_init=ms_list, precisions_init=ps_list, random_state=1234)
            gm.fit(np.concatenate([train3,test3],axis = 0))
            
            hh = pd.DataFrame(gm.predict_proba(train3), columns = cc_list[j])
            
            exp_cluster_size = 0.975*len(train3)/nc
            class_1 = (pd.DataFrame([yf]*nc).transpose().values * hh.values).sum(0)
            class_0 = (pd.DataFrame([1-yf]*nc).transpose().values * hh.values).sum(0)
            class_div = (0.5*np.abs(class_0[class_0 > 0.5*exp_cluster_size] - exp_cluster_size).mean()                       + 0.5*np.abs(class_1[class_1 > 0.5*exp_cluster_size] - exp_cluster_size).mean())/exp_cluster_size
            
            res = pd.concat([hh, yf.to_frame().reset_index(drop=True)], sort=False, axis=1)
            
            cluster_to_class = res.groupby('target').agg('mean').values.argmax(0)
            cluster_cnt[i,j,k] = cluster_to_class.sum()
            
            res = pd.concat([hh, pd.DataFrame(cluster_to_class, index=cc_list[j], 
                                              columns=['target']).transpose()], sort=False, axis=0).\
                transpose().groupby('target').agg(sum).transpose()
            
            cluster_div[i,j,k] = class_div
            res_list[j].append(res[1])
            gm_list[j].append(gm)
            ctc_list[j].append(cluster_to_class)
            acc[j,k] = -class_div
            all_acc[i,j,k] = (res.values.argmax(1) == yf.values).mean()
            all_roc[i,j,k] = roc_auc_score(yf.values, res[1])
    
    best_j = acc.mean(1).argmax()
    j_selection[best_j] += 1
    
    for k in np.argsort(acc[best_j,:])[-N_TEST:]:
        res2 = pd.concat([pd.DataFrame(gm_list[best_j][k].predict_proba(test3), columns = cc_list[best_j]), 
                          pd.DataFrame(ctc_list[best_j][k], index=cc_list[best_j], 
                                       columns=['target']).transpose()], sort=False, axis=0).\
            transpose().groupby('target').agg(sum).transpose()
        
        oof[idx1] += res_list[best_j][k]/N_TEST
        preds[idx2] += res2[1]/N_TEST
    
    if i%10==0: print('QMM scores CV =',round(roc_auc_score(train['target'],oof),5))

# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print(j_selection)
print('Final QMM scores CV =',round(auc,5))


# In[ ]:


for j in range(N_CLUST_OPT):
    print(np.all(cluster_cnt[:,j,:] == 0.5*nc_list[j]))


# # Submit Predictions

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)

import matplotlib.pyplot as plt
plt.hist(preds,bins=100)
plt.title('Final Test.csv predictions')
plt.show()

