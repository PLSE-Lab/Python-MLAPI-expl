#!/usr/bin/env python
# coding: utf-8

# **In this script we are using semisupervised, LabelSpreading Algorithm for prediction.**

# In[ ]:


import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.semi_supervised import LabelSpreading
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


oof_train = np.zeros(len(train)) 
pred_te_test = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# In[ ]:


for i in tqdm_notebook(range(512)):
    if (i==0):
        print("Starting the modelling")
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    
    y = train2.loc[idx1]['target']
    train2.reset_index(drop=True,inplace=True)

    sel = VarianceThreshold(threshold=1.5)
    train3 = sel.fit_transform(train2[cols])  
    test3 = sel.transform(test2[cols])

    skf = StratifiedKFold(n_splits=7, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        clf = QuadraticDiscriminantAnalysis(reg_param=0.25)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof_train[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        pred_te_test[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
    
print("ROC for training = ",roc_auc_score(train['target'],oof_train))


# In[ ]:


saved_targets = train['target'].values.copy()


# In[ ]:


test["target"] = pred_te_test


# In[ ]:


oof_ls = np.zeros(len(train)) 
pred_te_ls = np.zeros(len(test))


# In[ ]:


for k in tqdm_notebook(range(512)):
    train2 = train[train['wheezy-copper-turtle-magic']==k] 
    train2p = train2.copy(); idx1 = train2.index 
    test2 = test[test['wheezy-copper-turtle-magic']==k]
    
    test2p = test2[ (test2['target']<=0.001) | (test2['target']>=0.999) ].copy()
    test2p.loc[ test2p['target']>=0.5, 'target' ] = 1
    test2p.loc[ test2p['target']<0.5, 'target' ] = 0 
    train2p = pd.concat([train2p,test2p],axis=0)
    train2p.reset_index(drop=True,inplace=True)
    
    test2["target"] = -1
    #merging train2p with full test
    train3p = pd.concat([train2p,test2],axis=0)
    train3p.reset_index(drop=True,inplace=True)
    
    sel = VarianceThreshold(threshold=1.5).fit(train3p[cols])     
    train4p = sel.transform(train3p[cols])
    train4 = sel.transform(train2[cols])
    test4 = sel.transform(test2[cols])
    
    skf = StratifiedKFold(n_splits=25, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train4p, train3p['target']):
        test_index3 = test_index[ test_index<len(train4) ] 
        
        clf = LabelSpreading(gamma=0.01,kernel='rbf', max_iter=10)
        clf.fit(train4p[train_index,:],train3p.loc[train_index]['target'])
        oof_ls[idx1[test_index3]] = clf.predict_proba(train4[test_index3,:])[:,1]
        pred_te_ls[test2.index] += clf.predict_proba(test4)[:,1] / skf.n_splits

auc = roc_auc_score(saved_targets,oof_ls)
print('CV for LabelSpreading =',round(auc,5))               


# In[ ]:


#submission
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = pred_te_ls
sub.to_csv('submission.csv', index=False)

