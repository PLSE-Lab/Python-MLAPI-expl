#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# Before I get started, I just wanted to say: huge props to Inversion! The official starter kernel is **AWESOME**; it's so simple, clean, straightforward, and pragmatic. It certainly saved me a lot of time wrangling with data, so that I can directly start tuning my models (real data scientists will call me lazy, but hey I'm an engineer I just want my stuff to work).
# 
# I noticed two tiny problems with it:
# * It takes a lot of RAM to run, which means that if you are using a GPU, it might crash as you try to fill missing values.
# * It takes a while to run (roughly 3500 seconds, which is more than an hour; again, I'm a lazy guy and I don't like waiting).
# 
# With this kernel, I bring some small changes:
# * Decrease RAM usage, so that it won't crash when you change it to GPU. I simply changed when we are deleting unused variables.
# * Decrease **running time from ~3500s to ~40s** (yes, that's almost 90x faster), at the cost of a slight decrease in score. This is done by adding a single argument.
# 
# Again, my changes are super minimal (cause Inversion's kernel was already so awesome), but I hope it will save you some time and trouble (so that you can start working on cool stuff).
# 

# In[ ]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA,SparsePCA,KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


# # Efficient Preprocessing
# 
# This preprocessing method is more careful with RAM usage, which avoids crashing the kernel when you switch from CPU to GPU.

# In[ ]:


train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)

y = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity

# Drop target, fill in NaNs
train = train.drop('isFraud', axis=1)

train = train.fillna(-999)
test = test.fillna(-999)

# Label Encoding
for f in train.columns:
    if train[f].dtype=='object' or test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))   


# In[ ]:



#PCA/ICA for dimensionality reduction

n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

# # PCA
# pca = PCA(n_components=n_comp, random_state=420)
# pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
# pca2_results_test = pca.transform(test)
#

#Polynomial features
# poly = PolynomialFeatures(degree=1)
# poly_results_train = poly.fit_transform(train.drop(["y"], axis=1))
# poly_results_test = poly.transform(test)

#sparse PCA
spca = SparsePCA(n_components=n_comp, random_state=420)
spca2_results_train = spca.fit_transform(train)
spca2_results_test = spca.transform(test)

#Kernel PCA
# kpca = KernelPCA(n_components=n_comp, random_state=420)
# kpca2_results_train = kpca.fit_transform(train)
# kpca2_results_test = kpca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train)
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)



# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    # train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    # test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    # train['poly_' + str(i)] = poly_results_train[:, i - 1]
    # test['poly_' + str(i)] = poly_results_test[:, i - 1]

    train['spca_' + str(i)] = spca2_results_train[:, i - 1]
    test['spca_' + str(i)] = spca2_results_test[:, i - 1]

#     train['kpca_' + str(i)] = kpca2_results_train[:, i - 1]
#     test['kpca_' + str(i)] = kpca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]


print("After PCA/ICA")
print (len(list(train)))
print (len(list(test)))


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=42)


# # Training
# 
# DAYS OF RESEARCH BROUGHT ME TO THE CONCLUSION THAT I SHOULD SIMPLY SPECIFY `tree_method='gpu_hist'` IN ORDER TO ACTIVATE GPU (okay jk, took me an hour to figure out, but I wish XGBoost documentation was more clear about that).

# In[ ]:


clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)


# In[ ]:


get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# In[ ]:


y_test_pred = clf.predict(X_test)

score = roc_auc_score(y_test,y_test_pred )
print   ("score is {} ".format(score))


# In[ ]:


del X_train, X_test, y_train, y_test


# Some of you must be wondering how we were able to decrease the fitting time by that much. The reason for that is not only we are running on gpu, but we are also computing an approximation of the real underlying algorithm (which is a greedy algorithm). This hurts your score slightly, but as a result is much faster.
# 
# So why am I not using CPU with `tree_method='hist'`? If you try it out yourself, you'll realize it'll take ~ 7 min, which is still far from the GPU fitting time. Similarly, `tree_method='gpu_exact'` will take ~ 4 min, but likely yields better accuracy than `gpu_hist` or `hist`.
# 
# The [docs on parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) has a section on `tree_method`, and it goes over the details of each option.

# In[ ]:


sample_submission['isFraud'] = clf.predict_proba(test)[:,1]
sample_submission.to_csv('xgboost_v1_{}.csv'.format(score))

