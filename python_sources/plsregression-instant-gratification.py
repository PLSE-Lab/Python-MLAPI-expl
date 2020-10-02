#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, os
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


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
    
    
    X_train = train2[cols].values
    X_test = test2[cols].values
    
    poly = PolynomialFeatures(2)
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)
    
    y_train = train2['target'].values
    
    # STRATIFIED K-FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(X_train, y_train):
        
        # MODEL AND PREDICT WITH PLSR
        clf = PLSRegression(copy=True, max_iter=10000, n_components=512, scale=True,
        tol=3e-04)
        clf.fit(X_train[train_index], y_train[train_index])
        oof[idx1[test_index]] = np.hstack(clf.predict(X_train[test_index]))
        preds[idx2] += np.hstack(clf.predict(X_test)) / skf.n_splits
       
    if i%16==0: print(i)
        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print('PLSR scores CV =',round(auc,5))


# In[ ]:


sub = pd.read_csv("../input/sample_submission.csv")
sub["target"] = preds

sub.to_csv("submission.csv",index=False)

import matplotlib.pyplot as plt
plt.hist(preds,bins=100)
plt.title("Final Test.csv predictions")
plt.show()


# In[ ]:


sub.head(10)


# # remark

# In[ ]:




