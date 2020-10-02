#!/usr/bin/env python
# coding: utf-8

# # Objective
# In this notebook we will compare the public test set that Vlad released that has 0.973 LB for public test data only. We compare it with a standard QDA model and observe some interesting things.
# 
# The infamous "don't fork" kernel : https://www.kaggle.com/speedwagon/no-don-t-fork-it
# 
# Some analysis of the data was done here: https://www.kaggle.com/c/instant-gratification/discussion/94785#latest-547646
# It is theorized that the public test predictions are legitimate.

# In[ ]:


import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import svm, neighbors, linear_model, neural_network
from sklearn.svm import NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import pickle
import matplotlib.pylab as plt


# In[ ]:


train = pd.read_csv('../input/instant-gratification/train.csv')
test = pd.read_csv('../input/instant-gratification/test.csv')
with open('../input/predictions/prediction.pkl', 'rb') as f:
    vlad_preds = pickle.load(f)


# # Train a QDA model for comparison

# In[ ]:


oof_qda = np.zeros(len(train)) 
pred_te_qda = np.zeros(len(test))
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
oof = np.zeros(len(train))
preds = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for i in range(512):
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
        
        # MODEL AND PREDICT WITH QDA
        clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
       
    
auc = roc_auc_score(train['target'],oof)
print('QDA scores CV =',round(auc,5))


# # Plot Vlad's Predictions vs. QDA
# - Predictions are much more confident for target of 1
# - Less confident for target of 0

# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
test['qda_preds_target'] = preds
test['vlads_preds'] = vlad_preds[0:131073]
test[['qda_preds_target','vlads_preds']].sort_values('qda_preds_target').reset_index(drop=True)     .plot(style='.', alpha=0.1,
          title='Vlad 0.973 Public Test predictions vs. QDA - Ordered by QDA',
          ax=ax1)
test[['qda_preds_target','vlads_preds']].sort_values('vlads_preds').reset_index(drop=True)     .plot(style='.', alpha=0.1,
          title='Vlad 0.973 Public Test predictions vs. QDA - Ordered by Vlads',
          ax=ax2)
plt.show()


# # Distribution of the difference between Vlad and QDA

# In[ ]:


test['diff'] = test['vlads_preds'] - test['qda_preds_target']
test['diff'].plot(kind='hist', figsize=(15, 5), bins=200, title='Distribution of difference between Vlad and Simple QDA preds')
plt.show()


# In[ ]:


test['qda_preds_target'] = test['qda_preds_target'].round(5)


# In[ ]:


test[['vlads_preds','qda_preds_target','diff']].tail()


# In[ ]:


test[['vlads_preds','qda_preds_target','diff']]     .sort_values('diff')     .reset_index(drop=True)     .plot(style='.', figsize=(15, 5), title='Plot Predictions sorted by difference')
plt.show()


# In[ ]:


test.plot(x='vlads_preds',
          y='qda_preds_target',
          kind='scatter',
          figsize=(15, 15),
          alpha=0.2,
          title='Vlad Predictions vs QDA')
plt.show()


# # Plot by Rank

# In[ ]:


test['vlads_rank'] = test['vlads_preds'].rank(method='first')
test['qda_rank'] = test['qda_preds_target'].rank(method='first')
test.plot(x='vlads_rank', y='qda_rank', kind='scatter', figsize=(15, 15), alpha=0.2, title='Vlads public test preds vs QDA by Rank')
plt.show()

