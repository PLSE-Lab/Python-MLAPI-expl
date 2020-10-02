#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.decomposition import PCA
from tqdm import tqdm
import os
print(os.listdir("../input"))
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.feature_selection import VarianceThreshold


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# In[ ]:


def instant_model(train, test, cols = cols, clf = KNeighborsClassifier(leaf_size=2000, n_neighbors = 11, p = 2, weights = 'distance'), selection = "PCA"):
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))

    for i in tqdm(range(512)):

        train2 = train[train['wheezy-copper-turtle-magic'] == i]
        test2 = test[test['wheezy-copper-turtle-magic'] == i]
        idx1 = train2.index
        idx2 = test2.index

        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
        
        if selection == "variance":
            # StandardScaler & Variance selection
            data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=2).fit_transform(data[cols]))
            train3 = pd.DataFrame(data2[:train2.shape[0]], index = idx1)
            test3 = pd.DataFrame(data2[train2.shape[0]:], index = idx2)
            
        elif selection == "PCA":
            # PCA
            pca = PCA(n_components = 40, random_state= 1234)
            pca.fit(data[:train2.shape[0]])
            train3 = pd.DataFrame(pca.transform(data[:train2.shape[0]]), index = idx1)
            test3 = pd.DataFrame(pca.transform(data[train2.shape[0]:]), index = idx2)
        
        train3['target'] = train2['target']

        # Kfold
        skf = StratifiedKFold(n_splits=15, random_state=42)
        for train_index, test_index in skf.split(train3, train3['target']):
            # clf
            clf = clf
            X_train = train3.iloc[train_index, :].drop(["target"], axis = 1)
            X_test = train3.iloc[test_index, :].drop(["target"], axis = 1)
            y_train = train3.iloc[train_index, :]['target']
            y_test = train3.iloc[test_index, :]['target']
            clf.fit(X_train, y_train)

            # output
            train_prob = clf.predict_proba(X_train)[:,1]
            test_prob = clf.predict_proba(X_test)[:,1]
            oof[idx1[test_index]] = test_prob

            # bagging
            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
            # print("Chunk {0} Fold {1}".format(i, roc_auc_score(y_test, test_prob)))

    auc = roc_auc_score(train['target'], oof)
    print(f'AUC: {auc:.5}')
    
    return oof, preds, auc


# In[ ]:


def get_newtrain(train, test, preds, oof):
    # get useful train set from train and test data
    # get useful test 
    test['target'] = preds
    test.loc[test['target'] > 0.985, 'target'] = 1
    test.loc[test['target'] < 0.015, 'target'] = 0
    usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]
    print("No. of test records added: {}".format(usefull_test.shape[0]))
    # get useful train 
    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)
    new_train.loc[oof > 0.985, 'target'] = 1
    new_train.loc[oof < 0.015, 'target'] = 0
    return new_train


# In[ ]:


oof, preds, auc = instant_model(train, test, clf = QuadraticDiscriminantAnalysis(0.5),selection="variance")


# In[ ]:


newtrain = get_newtrain(train, test, preds, oof)
oof_qd, preds_qd, auc = instant_model(newtrain, test, clf = QuadraticDiscriminantAnalysis(0.5),selection="variance")


# In[ ]:


for i in range(1):
    newtrain = get_newtrain(newtrain, test, preds_qd, oof_qd)
    oof_qd, preds_qd, auc = instant_model(newtrain, test, clf = QuadraticDiscriminantAnalysis(0.5),selection="variance")


# In[ ]:





# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds_qd
sub.to_csv('submission.csv',index=False)


# In[ ]:




