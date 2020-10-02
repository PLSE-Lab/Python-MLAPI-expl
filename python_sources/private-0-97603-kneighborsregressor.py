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
import sympy 


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# In[ ]:


from sklearn.covariance import GraphicalLasso, OAS

def get_mean_cov(x,y):
    model = OAS()
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





# In[ ]:


from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from tqdm import tqdm_notebook

# INITIALIZE VARIABLES
cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
oof = np.zeros(len(train))
preds = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for i in tqdm_notebook(range(512)):
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
        
        brc = Birch(branching_factor=50, n_clusters=3, threshold=0.4, compute_labels=True)
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
        
        
#         ms, ps = get_mean_cov(x_train, y_train)
        
        gm = GaussianMixture(n_components=6, init_params='random', 
                             covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1, means_init=ms, precisions_init=ps)
        gm.fit(np.concatenate([train3[train_index,:],test3],axis = 0))
        oof[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:, 0:3].mean(axis=1)
        preds[idx2] += gm.predict_proba(test3)[:, 0:3].mean(axis=1) / skf.n_splits
    print('AUC ', i, roc_auc_score(1- train2['target'], oof[idx1]))    

        
# PRINT CV AUC
auc = roc_auc_score(1 - train['target'],oof)
print('QDA scores CV =',round(auc,5))


# In[ ]:


auc = roc_auc_score(1 - train['target'],oof)
print('QDA scores CV =',round(auc,5))


# In[ ]:


x_test_0 = pd.read_csv('../input/test.csv')
x_test_0['target']=preds


# In[ ]:


cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')


# In[ ]:


import numpy as np, pandas as pd, os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn import svm, neighbors

oof = np.zeros(len(train))
preds = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for k in tqdm_notebook(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==k] 
    train2p = train2.copy(); idx1 = train2.index 
    test2 = x_test_0[x_test_0['wheezy-copper-turtle-magic']==k]
    
    # ADD PSEUDO LABELED DATA
    test2p = test2
    train2p = pd.concat([train2p,test2p],axis=0)
    
    train2p.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     
    train3p = sel.transform(train2p[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
        
    # STRATIFIED K FOLD
    skf = KFold(n_splits=17, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3p):
        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof
        
#         print(train3p[train_index,:].shape)
        # MODEL AND PREDICT WITH QDA
        clf = neighbors.KNeighborsRegressor(n_neighbors=9, weights='distance')
        clf.fit(train3p[train_index,:], train2p.loc[train_index]['target'])
        oof[idx1[test_index3]] = clf.predict(train3[test_index3,:])
        preds[test2.index] += clf.predict(test3) / skf.n_splits
       
    if k%64==0: print(k)
        
# PRINT CV AUC
auc = roc_auc_score(train['target'], oof)
print('Pseudo Labeled QDA scores CV =',round(auc,5))


# In[ ]:





# In[ ]:





# In[ ]:





# # Submit Predictions

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)

import matplotlib.pyplot as plt
plt.hist(preds,bins=100)
plt.title('Final Test.csv predictions')
plt.show()


# ![](http://)
