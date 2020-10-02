#!/usr/bin/env python
# coding: utf-8

# # Lasso + Gaussian Mixture Models
# With this kernel I want to demonstrate how to use Gaussian mixture Models (GMM) which have the nice property to train unsupervised, so you can also use the test set. I use Graphical Lasso as an estimator for the initial value of precision matrix (= inverse Covariance) and mean

# In[ ]:


import numpy as np, pandas as pd, os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sympy 
import pickle
train = pd.read_csv('../input/instant-gratification/train.csv')
test = pd.read_csv('../input/instant-gratification/test.csv')

train.head()
train_main.head()


# In[ ]:


from sklearn.covariance import GraphicalLasso

def get_mean_cov(x,y):
    model = GraphicalLasso()
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


# # Estimate cov and mean from Lasso and predict with GMM

# In[ ]:


from sklearn.mixture import GaussianMixture

# INITIALIZE VARIABLES
cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
oof = np.zeros(len(train))
preds = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for i in tqdm(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    
    ms, ps = get_mean_cov(train3,train2['target'].values)
    
    # STRATIFIED K-FOLD
    skf = StratifiedKFold(n_splits=15, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        # MODEL AND PREDICT WITH QDA
        P = train3[train_index,:]
        T = train2.loc[train_index]['target'].values
        
        gm = GaussianMixture(n_components=2, init_params='kmeans', covariance_type='full', tol=0.1,reg_covar=0.1, max_iter=150, n_init=5,means_init=ms, precisions_init=ps)
        gm.fit(np.concatenate([P,test3],axis = 0))
        oof[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:,0]
        preds[idx2] += gm.predict_proba(test3)[:,0] / skf.n_splits

        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print('QDA scores CV =',round(auc,5))


# # Submit Predictions

# In[ ]:


sub = pd.read_csv('../input/instant-gratification/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)


# 
