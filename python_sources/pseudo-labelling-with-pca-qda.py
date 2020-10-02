#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# So far QDA with pseudo labelling has returned the best results in this competition. Pseudo labelling is used as a technique to augment the data set and improve the quality of the QDA fit.
# 
# **What will be the score if we slightly boost the accuracy of the pseudo labelled data with intermediate models (one or more)?**. 
# 
# Here I am using PCA-QDA as intermediate model (feel free to play with different models). My final prediction is 50 percent of PCA-QDA (pseudo label from variance-QDA) and 50 percent of Variance-QDA (pseudo label from PCA-QDA intermediate model).
# 
# 95% of the kernel is stolen from 
# [Dieter](https://www.kaggle.com/christofhenkel/lets-implement-qda-by-ourself) and [Chris](https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969). 

# # Intro
# Luckily Chris already put some illustrative pictures in his kernel, I can steal.
# Here is a pictorial explanation using sythetic 2D data. 
#   
# ## Step 1 - Build first model
# Given 50 training observations (25 target=1 yellow points, 25 target=0 blue points) we can estimate the multivariate (approx 40 dimensions) normal distributions of each of the two target types (0 & 1) by calculating empiral covariance and mean (see np.cov and np.mean) and then calculate that a given datapoint belongs to distribution A or B using scipy.stats.multivariate.
# 
# 
# ![image](http://playagricola.com/Kaggle/p16419.png)
# 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html
# 

# 
# ## Load Data

# In[ ]:


import numpy as np, pandas as pd, os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import NuSVC
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# 

# In[ ]:


#train['wheezy-copper-turtle-magic']


# In[ ]:


# INITIALIZE VARIABLES
cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')
oof = np.zeros(len(train))
preds = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for i in range(512):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(data[cols])
    data2 = sel.transform(data[cols])
    
    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]
    #sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    #train3 = sel.transform(train2[cols])
    #test3 = sel.transform(test2[cols])
    
    # STRATIFIED K-FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        # MODEL AND PREDICT WITH QDA
        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
        
        
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
       
    #if i%64==0: print(i)
        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print('QDA scores CV =',round(auc,5))


# #  Add pseudo label data and run an intermediate model (PCA-QDA) 

# I am using the variance threshold to estimate the number of components

# In[ ]:




cat_dict = dict()

# INITIALIZE VARIABLES
cols = [c for c in train.columns if c not in ['id', 'target']]
cols.remove('wheezy-copper-turtle-magic')

for i in range(512):

    
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    
    
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
        
    cat_dict[i] = train3.shape[1]


# The number of variables generated from the variance based threshold is not uniform. Therefore  I have chose to adapt the number of components in PCA in consequence.  

# In[ ]:



pd.DataFrame(list(cat_dict.items()))[1].value_counts().plot.barh()


# Same as in the original kernel I use pseudlabelling ...

# In[ ]:





# # Add pseudo label data from PCA-QDA and run a new model Variance-QDA

# In[ ]:



# INITIALIZE VARIABLES
test['target'] = preds
oof_var = np.zeros(len(train))
preds_var = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for k in range(512):
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
    
  
    
    pca = PCA(n_components=cat_dict[k], random_state= 1234)
    pca.fit(train2p[cols])
    train3p = pca.transform(train2p[cols])
    train3 = pca.transform(train2[cols])
    test3 = pca.transform(test2[cols])

           
        
    # STRATIFIED K FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3p, train2p['target']):
        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof
        
        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])
        oof_var[idx1[test_index3]] += clf.predict_proba(train3[test_index3,:])[:,1]
        preds_var[test2.index] += clf.predict_proba(test3)[:,1] / skf.n_splits
       
       
    #if k%64==0: print(k)
        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof_var)
print('Pseudo Labeled QDA scores CV =',round(auc,5)) #0.97035


# In[ ]:


# INITIALIZE VARIABLES
test['target'] = preds_var  
oof_var2 = np.zeros(len(train))
preds_var2 = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for k in range(512):
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
    
    
    
       
    sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     
    train3p = sel.transform(train2p[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
           
        
    # STRATIFIED K FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3p, train2p['target']):
        test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof
        
        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
        clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])
        oof_var2[idx1[test_index3]] += clf.predict_proba(train3[test_index3,:])[:,1]
        preds_var2[test2.index] += clf.predict_proba(test3)[:,1] / skf.n_splits
       
       
    #if k%64==0: print(k)
        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof_var2)
print('Pseudo Labeled QDA scores CV =',round(auc,5))


# # Final model accuracy

# In[ ]:


auc = roc_auc_score(train['target'],0.5*(oof_var+ oof_var2) )
print('Pseudo Labeled QDA scores CV =',round(auc,5))


# # Submit Predictions

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = 0.5* preds_var + 0.5*preds_var2
sub.to_csv('submission.csv',index=False)

import matplotlib.pyplot as plt
plt.hist(preds,bins=100)
plt.title('Final Test.csv predictions')
plt.show()


# # Conclusion
# **Let  try  intermediate models on pseudo labels?**
