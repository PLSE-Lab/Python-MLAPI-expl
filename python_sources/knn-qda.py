#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from tqdm import tqdm_notebook
import warnings
import multiprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

from sklearn.svm import NuSVC
from scipy.optimize import minimize  
warnings.filterwarnings('ignore')


# In[ ]:


train1 = pd.read_csv('../input/train.csv')
test1 = pd.read_csv('../input/test.csv')
cols = [c for c in train1.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


# In[ ]:


def instant_model(train, test, col, clf = QuadraticDiscriminantAnalysis(0.5)):
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    
    for i in tqdm_notebook(range(512)):

        train2 = train[train['wheezy-copper-turtle-magic'] == i]
        test2 = test[test['wheezy-copper-turtle-magic'] == i]
        idx1 = train2.index
        idx2 = test2.index

        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
        # feature selection
        data2 = VarianceThreshold(threshold = 2).fit_transform(data[cols])

        train3 = pd.DataFrame(data2[:train2.shape[0]], index = idx1)
        train3['target'] = train2['target']
        test3 = pd.DataFrame(data2[train2.shape[0]:], index = idx2)

        # Kfold
        skf = StratifiedKFold(n_splits=11, random_state=42)
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
    
    return oof, preds


# In[ ]:


def get_newtrain(train, test, preds, oof):
    # get useful train set from train and test data
    # get useful test 
    test['target'] = preds
    test.loc[test['target'] > 0.985, 'target'] = 1
    test.loc[test['target'] < 0.015, 'target'] = 0
    usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]

    # get useful train 
    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)
    new_train.loc[oof > 0.985, 'target'] = 1
    new_train.loc[oof < 0.015, 'target'] = 0
    return new_train


# In[ ]:


oof_temp, preds_temp = instant_model(train1, test1, cols, clf = QuadraticDiscriminantAnalysis(0.5))


# In[ ]:


newtrain1 = get_newtrain(train1, test1, preds_temp, oof_temp)
cols1 = [c for c in newtrain1.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]


oof_qda, preds_qda = instant_model(newtrain1, test1,cols1)
oof_knn, preds_knn = instant_model(newtrain1, test1,cols1, clf = KNeighborsClassifier(n_neighbors = 7, p = 2, weights = 'distance'))


# Stacking QDA and KNN

# In[ ]:


log = LogisticRegression()

final_feature = pd.DataFrame({'QDA':oof_qda, 'KNN':oof_knn})
final_feature_test = pd.DataFrame({'QDA':preds_qda, 'KNN':preds_knn})

y = newtrain1.target
log.fit(final_feature, y)
print(roc_auc_score(newtrain1['target'], log.predict_proba(final_feature)[:,1]))

preds = log.predict_proba(final_feature_test)[:,1]


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission_stack.csv',index=False)

