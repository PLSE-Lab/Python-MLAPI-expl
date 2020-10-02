#!/usr/bin/env python
# coding: utf-8

# # PRIIVATE LB 0.97564 33th place

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")
# Generating data ----
import numpy as np, pandas as pd, os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import NuSVC
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# # The model

# ## Kmeans to initialize gaussian mixture model

# Since the target are divided into 2 labels 0 and 1 and made from n gaussians. We tried to cluster each label (0 and 1) to k clusters and we tuned this param k depending on the validation score. We calculate the precision and location for each cluster to initialize gaussian mixture model.
# The k we used here is 4.

# In[ ]:


from sklearn.covariance import GraphicalLasso
from sklearn.cluster import KMeans
from sklearn.covariance import ShrunkCovariance,EmpiricalCovariance,LedoitWolf
def get_mean_cov(x,y):
    model =  LedoitWolf()
    ones = (y==1).astype(bool)
    x2 = x[ones]
    #K=4
    kmeans = KMeans(4, random_state=125,n_jobs=4).fit(x2)
    kmeans_predictions = (kmeans.predict(x2))
    #Calculating the precision and the location of cluster 0 of the 1 labels.
    onesb = (kmeans_predictions==0).astype(bool)
    x21 = x2[onesb]
    model.fit(x21)
    p1 = model.precision_
    m1 = model.location_
    #Calculating the precision and the location of cluster 1 of the 1 labels.
    onesb = (kmeans_predictions==1).astype(bool)
    x22 = x2[onesb]
    model.fit(x22)
    p2 = model.precision_
    m2 = model.location_
    #Calculating the precision and the location of cluster 2 of the 1 labels.
    onesb = (kmeans_predictions==2).astype(bool)
    x23 = x2[onesb]
    model.fit(x23)
    p3 = model.precision_
    m3 = model.location_
    #Calculating the precision and the location of cluster 3 of the 1 labels.
    onesb = (kmeans_predictions==3).astype(bool)
    x24 = x2[onesb]
    model.fit(x24)
    p4 = model.precision_
    m4 = model.location_
    
  
    
    onesb = (y==0).astype(bool)
    x2b = x[onesb]
    #K for labels 0 is 4 too
    kmeans = KMeans(4, random_state=125,n_jobs=4).fit(x2b)
    kmeans_predictions = (kmeans.predict(x2b))
    #Calculating the precision and the location of cluster 0 of the 0 labels.
    onesb = (kmeans_predictions==0).astype(bool)
    x2b1 = x2b[onesb]
    model.fit(x2b1)
    p5 = model.precision_
    m5 = model.location_
    #Calculating the precision and the location of cluster 1 of the 0 labels.    
    onesb = (kmeans_predictions==1).astype(bool)
    x2b2 = x2b[onesb]
    model.fit(x2b2)
    p6 = model.precision_
    m6 = model.location_
    #Calculating the precision and the location of cluster 2 of the 0 labels.
    onesb = (kmeans_predictions==2).astype(bool)
    x2b3 = x2b[onesb]
    model.fit(x2b3)
    p7 = model.precision_
    m7 = model.location_
    #Calculating the precision and the location of cluster 3 of the 0 labels.
    onesb = (kmeans_predictions==3).astype(bool)
    x2b4 = x2b[onesb]
    model.fit(x2b4)
    p8 = model.precision_
    m8 = model.location_

    #Stacking of all the means and covariances  
    ms = np.stack([m1,m2,m3,m4,m5,m6,m7,m8])
    ps = np.stack([p1,p2,p3,p4,p5,p6,p7,p8])
    return ms,ps


# ## Gaussian mixture model

# In[ ]:


#SELECTING COLUMNS
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
    sel = VarianceThreshold(threshold=2).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
    
    # STRATIFIED K-FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        

        #Initialization of the mean and covariance of each cluster within labels 0 and 1 (4clusters for labels 1 and 4clusters for labels 0 )
        ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values)
        #Gaussian Mixture Modelling 
        gm = GaussianMixture(n_components=8, init_params='random', covariance_type='full', tol=0.001,reg_covar=1, max_iter=100, n_init=1,means_init=ms, precisions_init=ps)
        #Since learning with GMM is unsupervised we will fit the train+validation+test 
        gm.fit(np.concatenate([train3,test3],axis = 0))
        #The predict_proba will return (n_clusters=4+4=8 ,  num_samples)
        #We will predict the probabilty of being the class 1 and that represent the sum of the probabilities of the 4 first columns and that explains the [:,:4].sum(axis=1)
        oof[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:,:4].sum(axis=1)
        preds[idx2] += gm.predict_proba(test3)[:,:4].sum(axis=1) / skf.n_splits
    auc = roc_auc_score(train2['target'],oof[idx1])
    print('GMM scores CV =',round(auc,5))
        

        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print('GMM scores CV =',round(auc,5))


# ## pseudolabeling

# We used the pseudo labeling  proposed by [chris](https://www.kaggle.com/cdeotte) in [this](https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969) link

# In[ ]:


test['target'] = preds
oof_final6 = np.zeros(len(train))
final_preds = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for k in tqdm_notebook(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==k] 
    train2p = train2.copy(); idx1 = train2.index 
    test2 = test[test['wheezy-copper-turtle-magic']==k]
    
    # ADD PSEUDO LABELED DATA
    test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()
    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1
    test2p.loc[ test2p['target']<0.5, 'target' ] = 0 
    train2p = pd.concat([train2p,test2p],axis=0)
    train2p.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=2).fit(train2p[cols])     
    train3p = sel.transform(train2p[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
        
    # STRATIFIED K FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3p, train2p['target']):
        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof
        
        # MODEL AND PREDICT WITH GMM
        ms, ps = get_mean_cov(train3p[train_index,:],train2p.loc[train_index]['target'].values)
        
        gm = GaussianMixture(n_components=8, init_params='random', covariance_type='full', 
                             tol=0.001,reg_covar=1, max_iter=100, n_init=1,means_init=ms, precisions_init=ps)
        
        gm.fit(np.concatenate([train3p,test3],axis = 0))
        oof_final6[idx1[test_index3]] = gm.predict_proba(train3[test_index3,:])[:,:4].sum(axis=1)
        final_preds[test2.index] += gm.predict_proba(test3)[:,:4].sum(axis=1) / skf.n_splits
       
    #if k%64==0: print(k)
        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof_final6)
print('Pseudo Labeled GMM scores CV =',round(auc,5))


# # Ensembling 

# We averaged the prediction with  a simple GMM and the prediction with GMM using pseudo labeling 

# In[ ]:


auc = roc_auc_score(train['target'],oof_final6+oof)
print('Pseudo Labeled GMM scores CV =',round(auc,5))


# In[ ]:


sample = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


sample.target = (final_preds+preds)/2
sample.to_csv("submission.csv",index=False)


# In[ ]:


sample.head()


# In[ ]:




