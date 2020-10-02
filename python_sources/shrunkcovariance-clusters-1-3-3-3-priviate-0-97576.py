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


import numpy as np, pandas as pd, os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm


from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import sklearn.mixture as mixture
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.covariance
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, MinCovDet, LedoitWolf, OAS, ShrunkCovariance
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical
from scipy.stats import rankdata

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# In[ ]:


# Return clusters-list
def get_kmeans_clusters(x, y, k_pos=1, k_neg=1):
    
    x_zeros = x[y==0]
    x_ones  = x[y==1]

    model_0 = KMeans(n_clusters=k_neg)
    model_1 = KMeans(n_clusters=k_pos)

    model_0.fit(x_zeros)
    model_1.fit(x_ones)

    model_0_clus = [x_zeros[model_0.labels_==k] for k in range(model_0.n_clusters)]
    model_1_clus = [x_ones[model_1.labels_==k] for k in range(model_1.n_clusters)]

    return model_1_clus + model_0_clus

# Return gmm model
def fit_multicluster_gmm(x, y, xt, k_pos, k_neg, max_iter=100):
    """
    x = train predictors
    y = binary target
    xt = test predictors
    k_pos = number clusters when y==1
    k_neg = number clusters when y==0
    """

    clusters = get_kmeans_clusters(x, y, k_pos=k_pos, k_neg=k_neg)

    for i in range(len(clusters)):

        x_cluster = clusters[i]

        #if(x_cluster.shape[0]<x_cluster.shape[1]):
            #model = GraphicalLasso(mode='lars', max_iter=max_iter)
        #else:
        model = ShrunkCovariance()

        model.fit(x_cluster)

        if (i==0):
            ps = np.expand_dims(model.precision_, axis=0)
            ms = np.expand_dims(model.location_,  axis=0)
        else:
            ps = np.concatenate([ps, np.expand_dims(model.precision_, axis=0)], axis=0)
            ms = np.concatenate([ms, np.expand_dims(model.location_, axis=0)], axis=0)

    gm = mixture.GaussianMixture(n_components=k_pos+k_neg, 
                                 init_params='random', 
                                 covariance_type='full',
                                 tol=0.001,
                                 reg_covar=0.001, 
                                 max_iter=100,
                                 n_init=5,
                                 means_init=ms,
                                 precisions_init=ps)

    gm.fit(np.vstack((x.astype(np.float), xt.astype(np.float))))
    preds = gm.predict_proba(x.astype(np.float))[:,0]
    score = roc_auc_score(y, preds)
    return score, gm, k_pos, k_neg

def get_mean_cov(x,y, model=GraphicalLasso(), max_iter=100):
    
    try:
        model.set_params(**{'max_iter':200})
    except:
        pass
    
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


def extract_wheezy_copper_turtle_magic(train, i):
    train2 = train[train['wheezy-copper-turtle-magic']==i].copy()
        
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    target = train2['target'].astype(np.int).values
    
    train2.reset_index(drop=True, inplace=True)
    
    return train2.drop(['id', 'target'], axis=1), test2.drop(['id'], axis=1), idx1, idx2, target


# In[ ]:


def data_augmentation(X, y, gm_means, gm_dist):
    for clus in range(len(gm_means)):
        A = 2*gm_means[clus,].reshape(1,-1) - X[gm_dist==clus]
        B = X[gm_dist==clus]
        t = y[gm_dist==clus]
        if clus==0:
            Xn = np.vstack((A, B))
            yn = np.concatenate((t,t))
        else:
            Xn = np.vstack((Xn, A, B))
            yn = np.concatenate((yn, t,t))
    return Xn, yn
    


# In[ ]:


# Here we initialize variables
cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
oof = np.zeros(len(train))
test_preds = np.zeros(len(test))
oof_gmm = np.zeros(len(train))
test_preds_gmm = np.zeros(len(test))
oof_gmm_2 = np.zeros(len(train))
test_preds_gmm_2 = np.zeros(len(test))
trials = 3
cat_dict = dict()
cluster_report = list()

# BUILD 512 SEPARATE MODELS
for i in tqdm(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2, test2, idx1, idx2, target = extract_wheezy_copper_turtle_magic(train, i)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    
    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    #pipe = Pipeline([('vt', VarianceThreshold(threshold=1.5)), ('scaler', StandardScaler())])
    pipe = Pipeline([('vt', VarianceThreshold(threshold=1.5))])
   
    train3 = pipe.fit_transform(train2[cols])
    test3 = pipe.fit_transform(test2[cols])
    
    # We just record in cat_dict how many variables have a threshold above 1.5
    cat_dict[i] = train3.shape[1]
    
    # remove correlations between features
    svd = PCA(n_components=cat_dict[i])
    svd.fit(np.vstack((train3, test3)))
    train3 = svd.transform(train3)
    test3 = svd.transform(test3)
        

    ################# First GMM with 3 positive clusters and 3 negative clusters
    
    try:
        score, gm, k_pos, k_neg = fit_multicluster_gmm(x=train3, y=target, xt=test3, k_pos=3, k_neg=3)
    except:
        try:
            print("Falling back to k_pos/k_neg=2")
            score, gm, k_pos, k_neg = fit_multicluster_gmm(x=train3, y=target, xt=test3, k_pos=2, k_neg=2)
        except:
            print("Falling back to k_pos/k_neg=1")
            score, gm, k_pos, k_neg = fit_multicluster_gmm(x=train3, y=target, xt=test3, k_pos=1, k_neg=1)
    
    clusters = gm.predict_proba(train3).shape[1]
    oof_gmm[idx1] = np.sum(gm.predict_proba(train3)[:,:clusters//2], axis=1)
    test_preds_gmm[idx2] += np.sum(gm.predict_proba(test3)[:,:clusters//2], axis=1)
    
    ################# Try data augmentation and second GMM with 3 positive clusters and 3 negative clusters
    
    #train3, target = data_augmentation(train3, target, gm_means=gm.means_, gm_dist=gm.predict(train3))
    
    try:
        score, gm, k_pos, k_neg = fit_multicluster_gmm(x=train3, y=target, xt=test3, k_pos=1, k_neg=3)
    except:
        try:
            print("Falling back to k_pos/k_neg=2")
            score, gm, k_pos, k_neg = fit_multicluster_gmm(x=train3, y=target, xt=test3, k_pos=1, k_neg=2)
        except:
            print("Falling back to k_pos/k_neg=1")
            score, gm, k_pos, k_neg = fit_multicluster_gmm(x=train3, y=target, xt=test3, k_pos=1, k_neg=1)
    
    clusters = gm.predict_proba(train3).shape[1]
    oof_gmm_2[idx1] = gm.predict_proba(train3)[:,0]
    test_preds_gmm_2[idx2] += gm.predict_proba(test3)[:,0]


# In[ ]:


# PRINT CV AUC
oof_auc_gmm = roc_auc_score(train['target'], oof_gmm)
print('OOF AUC: =',round(oof_auc_gmm, 5))

oof_auc_gmm_2 = roc_auc_score(train['target'], oof_gmm_2)
print('OOF AUC: =',round(oof_auc_gmm_2, 5))

oof_auc_blend = roc_auc_score(train['target'], (oof_gmm+oof_gmm_2)/2)
print('OOF AUC: =',round(oof_auc_blend, 5))


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = test_preds_gmm
sub.to_csv('submission_gmm.csv',index=False)

sub['target'] = test_preds_gmm_2
sub.to_csv('submission_gmm_2.csv',index=False)

sub['target'] = (test_preds_gmm + test_preds_gmm_2)/2
sub.to_csv('submission_blend.csv',index=False)


# In[ ]:


import matplotlib.pyplot as plt
plt.hist(test_preds_gmm, bins=100)
plt.title('Final Test.csv predictions')
plt.show()


# In[ ]:


plt.hist(test_preds_gmm_2, bins=100)
plt.title('Final Test.csv predictions')
plt.show()


# In[ ]:


plt.hist((test_preds_gmm + test_preds)/2, bins=100)
plt.title('Final Test.csv predictions')
plt.show()

