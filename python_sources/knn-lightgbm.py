#!/usr/bin/env python
# coding: utf-8

# In this kernel, I'm presenting another approach to the problem, that I didn't see in the public kernels. (Maybe I have not noticed, sorry in that case).
# 
# The approach is to take the closest samples from 0 and 1 classes, for each querry sample, then extract some features from them, and train a classifier on top of those features. Here I have used LightGBM, and my features are normal statistical features calculated from the distance matrices.

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, os
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score as auc
from functools import partial

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
MagicFeat = 'wheezy-copper-turtle-magic'
cols = [c for c in train_df.columns if c not in ['id', 'target', MagicFeat]]


# This part is mainly taken from Chris's kernel:
# https://www.kaggle.com/cdeotte/support-vector-machine-0-925
# 
# Thanks [@cdeotte](https://www.kaggle.com/cdeotte)

# In[ ]:


def trainer(Model, train, test):
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    # BUILD 512 SEPARATE MODELS
    for i in range(512):
        # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
        train2 = train[train[MagicFeat]==i]
        test2 = test[test[MagicFeat]==i]
        idx1 = train2.index; idx2 = test2.index
        train2.reset_index(drop=True,inplace=True)

        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
        sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
        train3 = sel.transform(train2[cols])
        test3 = sel.transform(test2[cols])
        
        # STRATIFIED K-FOLD
        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
        for train_index, test_index in skf.split(train3, train2['target']):
            # Train MODEL AND PREDICT
            clf = Model()
            clf.fit(train3[train_index,:], train2.loc[train_index]['target'])
            oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
    return oof, preds


# The wrapper class of the KNNs and the classifier:

# In[ ]:


from sklearn.neighbors import NearestNeighbors
from lightgbm import LGBMClassifier
from scipy.stats import skew, kurtosis, hmean, gmean

def getFeatures(dists):
    '''
    Generates features from the distance matrix of shape (n_samples by n_neighbors)
    '''
    Features = []
    Funcs = [np.amin, np.amax, np.mean, np.std, np.median, hmean, gmean, kurtosis, skew]+                [partial(np.percentile, q=p) for p in [1,5,10,90,95,99]]
    for func in Funcs:
        Features.append(func(dists, axis=1).reshape(-1,1))
    return np.concatenate(Features, axis=1)


class KNN_CLF():
    '''
    Binary KNN Classifier:
    It takes the distances to the k nearest neighbors of a node from
    samples from both 0, and 1 classes, extracts some features from
    the distance matrices, and trains a classifier over it.
    By default the classifier is LightGBM.
    '''
    __slots__ = ['k', 'knn0', 'knn1', 'clf']
    def __init__(self, knn_params={}, k=5, CLF=LGBMClassifier, clf_params={}):
        self.k = k
        self.knn0 = NearestNeighbors(**knn_params)
        self.knn1 = NearestNeighbors(**knn_params)
        self.clf = CLF(**clf_params)
        
    def fit(self, X, y):
        self.knn0.fit(X[y==0, :])
        self.knn1.fit(X[y==1, :])
        # during training the first neighbor is the sample itself, 
        # so we take k+1 neighbors and take the first out
        F0 = getFeatures(
            self.knn0.kneighbors(X,
                                 n_neighbors=min(self.k+1, self.knn0._fit_X.shape[0]),
                                 return_distance=True)[0][:,1:])
        F1 = getFeatures(
            self.knn1.kneighbors(X,
                                 n_neighbors=min(self.k+1, self.knn1._fit_X.shape[0]),
                                 return_distance=True)[0][:,1:])
        XX = np.concatenate((F0, F1, F1-F0), axis=1)
        self.clf.fit(XX, y, verbose=0)

    def predict_proba(self, X):
        F0 = getFeatures(
            self.knn0.kneighbors(X,
                                 n_neighbors=min(self.k, self.knn0._fit_X.shape[0]),
                                 return_distance=True)[0])
        F1 = getFeatures(
            self.knn1.kneighbors(X,
                                 n_neighbors=min(self.k, self.knn1._fit_X.shape[0]),
                                 return_distance=True)[0])
        XX = np.concatenate((F0, F1, F1-F0), axis=1)
        return self.clf.predict_proba(XX)
        


# training ...

# In[ ]:


Model = partial(KNN_CLF, k=150)
trOut, tsOut = trainer(Model, train_df, test_df)
print(auc(train_df.target, trOut))

import matplotlib.pyplot as plt
plt.hist(trOut,bins=100)
plt.title('OOF predictions')
plt.show()

plt.figure()
plt.hist(tsOut,bins=100)
plt.title('Test.csv predictions')
plt.show()


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = tsOut
sub.to_csv('submission.csv',index=False)

