#!/usr/bin/env python
# coding: utf-8

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
from sklearn import linear_model
import sympy 
import numpy as np
from sklearn.covariance import OAS


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# In[ ]:





# In[ ]:


from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN

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
    
    # STRATIFIED K-FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        x_train, y_train = train3[train_index,:], train2.loc[train_index]['target'].values
        
        x_train_0 = x_train[y_train==0]
        x_train_1 = x_train[y_train==1]
        
#         brc = BayesianGaussianMixture(n_components=3, covariance_type='full', weight_concentration_prior=1e-2, 
#                                       weight_concentration_prior_type='dirichlet_process', mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
#                                       init_params="random", max_iter=100, random_state=666)#Birch(branching_factor=50, n_clusters=3, threshold=0.4, compute_labels=True)
        brc = Birch(branching_factor=50, n_clusters=3, threshold=0.6, compute_labels=True)
        labels_0 = brc.fit_predict(x_train_0)
        labels_1 = brc.fit_predict(x_train_1) 
        
        zero_mean = []
        zero_cov = []
        for l in np.unique(labels_0):
            model = OAS()
            model.fit(x_train_0[labels_0==l])
            p = model.precision_
            m = model.location_
            
            zero_mean.append(m)
            zero_cov.append(p)
            
        one_mean = []
        one_cov = []
        for l in np.unique(labels_1):
            model = OAS()
            model.fit(x_train_1[labels_1==l])
            p = model.precision_
            m = model.location_
            
            one_mean.append(m)
            one_cov.append(p)
       
            
            
        
#         print(np.array(zero_mean).mean(axis=0))
        
        ms = np.stack(zero_mean + one_mean)
        ps = np.stack(zero_cov +  one_cov)
        
      
        gm = GaussianMixture(n_components=6, init_params='kmeans', 
                             covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,
                             means_init=ms, precisions_init=ps, random_state=666)
        gm.fit(np.concatenate([train3[train_index,:],test3],axis = 0))
        oof[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:, 0:3].mean(axis=1)
        preds[idx2] += gm.predict_proba(test3)[:, 0:3].mean(axis=1) / skf.n_splits
    print('AUC ', i, roc_auc_score(1- train2['target'], oof[idx1]))    

        
# PRINT CV AUC
auc = roc_auc_score(1 - train['target'],oof)
print('QDA scores CV =',round(auc,5))


# In[ ]:





# In[ ]:


oof_0 = oof
preds_0 = preds


# In[ ]:





# In[ ]:


from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN

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
    
    # STRATIFIED K-FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        x_train, y_train = train3[train_index,:], train2.loc[train_index]['target'].values
        
        x_train_0 = x_train[y_train==0]
        x_train_1 = x_train[y_train==1]
        
#         brc = BayesianGaussianMixture(n_components=3, covariance_type='full', weight_concentration_prior=1e-2, 
#                                       weight_concentration_prior_type='dirichlet_process', mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
#                                       init_params="random", max_iter=100, random_state=666)#Birch(branching_factor=50, n_clusters=3, threshold=0.4, compute_labels=True)
        brc = Birch(branching_factor=50, n_clusters=4, threshold=0.6, compute_labels=True)
        labels_0 = brc.fit_predict(x_train_0)
        labels_1 = brc.fit_predict(x_train_1) 
        
        zero_mean = []
        zero_cov = []
        for l in np.unique(labels_0):
            model = OAS()
            model.fit(x_train_0[labels_0==l])
            p = model.precision_
            m = model.location_
            
            zero_mean.append(m)
            zero_cov.append(p)
            
        one_mean = []
        one_cov = []
        for l in np.unique(labels_1):
            model = OAS()
            model.fit(x_train_1[labels_1==l])
            p = model.precision_
            m = model.location_
            
            one_mean.append(m)
            one_cov.append(p)
       
            
            
        
#         print(np.array(zero_mean).mean(axis=0))
        
        ms = np.stack(zero_mean + one_mean)
        ps = np.stack(zero_cov +  one_cov)
        
      
        gm = GaussianMixture(n_components=8, init_params='kmeans', 
                             covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,
                             means_init=ms, precisions_init=ps, random_state=666)
        gm.fit(np.concatenate([train3[train_index,:],test3],axis = 0))
        oof[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:, 0:4].mean(axis=1)
        preds[idx2] += gm.predict_proba(test3)[:, 0:4].mean(axis=1) / skf.n_splits
    print('AUC ', i, roc_auc_score(1- train2['target'], oof[idx1]))    

        
# PRINT CV AUC
auc = roc_auc_score(1 - train['target'],oof)
print('QDA scores CV =',round(auc,5))


# In[ ]:


oof_1 = oof
preds_1 = preds


# In[ ]:





# In[ ]:


from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN

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
    
    # STRATIFIED K-FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        x_train, y_train = train3[train_index,:], train2.loc[train_index]['target'].values
        
        x_train_0 = x_train[y_train==0]
        x_train_1 = x_train[y_train==1]
        
#         brc = BayesianGaussianMixture(n_components=3, covariance_type='full', weight_concentration_prior=1e-2, 
#                                       weight_concentration_prior_type='dirichlet_process', mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
#                                       init_params="random", max_iter=100, random_state=666)#Birch(branching_factor=50, n_clusters=3, threshold=0.4, compute_labels=True)
        brc = Birch(branching_factor=50, n_clusters=2, threshold=0.6, compute_labels=True)
        labels_0 = brc.fit_predict(x_train_0)
        labels_1 = brc.fit_predict(x_train_1) 
        
        zero_mean = []
        zero_cov = []
        for l in np.unique(labels_0):
            model = OAS()
            model.fit(x_train_0[labels_0==l])
            p = model.precision_
            m = model.location_
            
            zero_mean.append(m)
            zero_cov.append(p)
            
        one_mean = []
        one_cov = []
        for l in np.unique(labels_1):
            model = OAS()
            model.fit(x_train_1[labels_1==l])
            p = model.precision_
            m = model.location_
            
            one_mean.append(m)
            one_cov.append(p)
       
            
            
        
#         print(np.array(zero_mean).mean(axis=0))
        
        ms = np.stack(zero_mean + one_mean)
        ps = np.stack(zero_cov +  one_cov)
        
      
        gm = GaussianMixture(n_components=4, init_params='kmeans', 
                             covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,
                             means_init=ms, precisions_init=ps, random_state=666)
        gm.fit(np.concatenate([train3[train_index,:],test3],axis = 0))
        oof[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:, 0:2].mean(axis=1)
        preds[idx2] += gm.predict_proba(test3)[:, 0:2].mean(axis=1) / skf.n_splits
    print('AUC ', i, roc_auc_score(1- train2['target'], oof[idx1]))    

        
# PRINT CV AUC
auc = roc_auc_score(1 - train['target'],oof)
print('QDA scores CV =',round(auc,5))


# In[ ]:


oof_2 = oof
preds_2 = preds


# In[ ]:





# In[ ]:


train['y_0'] = oof_0
test['y_0'] = preds_0

train['y_1'] = oof_1
test['y_1'] = preds_1

train['y_2'] = oof_2
test['y_2'] = preds_2


# In[ ]:


oof_features = ['y_0', 'y_1', 'y_2']


# In[ ]:





# In[ ]:


for m in sorted(train['wheezy-copper-turtle-magic'].unique()):
        idx_tr = (train['wheezy-copper-turtle-magic']==m)
        idx_te = (test['wheezy-copper-turtle-magic']==m)
        oofs = []
        preds = []
        kf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
        oof_preds = np.zeros((len(train[idx_tr][oof_features]), 1))
        test_preds = np.zeros((len(test[idx_te][oof_features]), 1))
        for idx, (train_index, valid_index) in enumerate(kf.split(train[idx_tr][oof_features], train[idx_tr]['target'])):
                y_train, y_valid = train[idx_tr]['target'].iloc[train_index], train[idx_tr]['target'].iloc[valid_index]
                x_train, x_valid = train[idx_tr][oof_features].iloc[train_index,:], train[idx_tr][oof_features].iloc[valid_index,:]

                model = linear_model.Ridge(alpha=3)
                model.fit(x_train, y_train)   
                oof_preds[valid_index, :] = model.predict(x_valid).reshape((-1, 1))
                test_preds += model.predict(test[idx_te][oof_features]).reshape((-1, 1)) / 11  
        print('OOF AUC ', m, ' ', roc_auc_score(train[idx_tr]['target'], oof_preds))
        oofs.append((idx_tr, oof_preds))
        preds.append((idx_te, test_preds))
        for ids, target in oofs:
            train.loc[ids,'target_final'] = target
        for ids, target in preds:
            test.loc[ids,'target_final'] = target
            
print('OOF AUC ', roc_auc_score(train['target'], train['target_final']))


# In[ ]:


test['target_final'].values


# In[ ]:





# In[ ]:


pd.DataFrame(np.concatenate([preds_0.reshape(-1, 1), preds_1.reshape(-1, 1), preds_2.reshape(-1, 1), 1 - test['target_final'].values.reshape(-1, 1)], axis=1)).rank().corr()


# In[ ]:


test_preds = pd.DataFrame(np.concatenate([preds_0.reshape(-1, 1), preds_1.reshape(-1, 1), preds_2.reshape(-1, 1), 1 - test['target_final'].values.reshape(-1, 1)], axis=1))


# In[ ]:


y_hat = test_preds.rank(axis=0, method='min').mul(test_preds.shape[1] * [1 / test_preds.shape[1]]).sum(1) / test_preds.shape[0]


# In[ ]:


y_hat.head()


# In[ ]:





# # Submit Predictions

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = 1 - test['target_final'].values
sub.to_csv('submission.csv',index=False)

import matplotlib.pyplot as plt
plt.hist(1 - test['target_final'].values ,bins=100)
plt.title('Final Test.csv predictions')
plt.show()

