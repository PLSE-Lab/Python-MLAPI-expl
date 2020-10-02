#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.mixture import GaussianMixture
from tqdm import tqdm_notebook
import pickle
import warnings

warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/instant-gratification/train.csv')
test = pd.read_csv('../input/instant-gratification/test.csv')
train.head()


# In[ ]:


model = pickle.load(open('../input/instant-model-v2/gmm_pretrained.v2.pkl','rb'))


# In[ ]:


# INITIALIZE VARIABLES
cols = [c for c in train.columns if c not in ['id', 'wheezy-copper-turtle-magic', 'target']]
cols_t = [c for c in train.columns if c not in ['id', 'wheezy-copper-turtle-magic']]
oof = np.zeros(len(train))
preds = np.zeros(len(test))
# BUILD 512 SEPARATE MODELS
for i in tqdm_notebook(range(512)):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    corr = train2[cols_t].corr()['target'].abs()
    corr_index = corr[corr > 0.01].drop(labels=['target']).index
    var = train2[cols].var()
    var_index = var[var > 2].index
    columns = var_index.intersection(corr_index)
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    #pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])
    pipe1 = Pipeline([('scaler', StandardScaler())])
    data1 = pd.concat([pd.DataFrame(train2[columns]), pd.DataFrame(test2[columns])])
    data1 = pipe1.fit_transform(data1[columns])
    train3 = data1[:train2.shape[0]]
    test3 = data1[train2.shape[0]:]
    train3_gm = model[i].predict_proba(train3)
    test3_gm = model[i].predict_proba(test3)
    train_label = model[i].predict(train3)
    test_label = model[i].predict(test3)
    clf = QuadraticDiscriminantAnalysis(reg_param=0.225)
    clf.fit(np.vstack((train3, test3)), np.concatenate((train_label, test_label)))
    train3_qda = clf.predict_proba(train3)
    test3_qda = clf.predict_proba(test3)
    train3 = np.hstack((train3_gm, train3_qda))
    test3 = np.hstack((test3_gm, test3_qda))
    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(train3, train2['target']): 
        x_train = train3[train_index,:]
        y_train = train2.loc[train_index]['target'].values
        x_val = train3[test_index,:]
        clf = QuadraticDiscriminantAnalysis(reg_param=0.075)
        clf.fit(x_train,y_train)
        oof[idx1[test_index]] = clf.predict_proba(x_val)[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print('CV scores = ',round(auc,5))


# In[ ]:


for itr in range(2):
    # INITIALIZE VARIABLES
    oof2 = np.zeros(len(train))
    preds = np.zeros(len(test))
    # BUILD 512 SEPARATE MODELS
    for i in tqdm_notebook(range(512)):
        # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
        train2 = train[train['wheezy-copper-turtle-magic']==i]
        test2 = test[test['wheezy-copper-turtle-magic']==i]
        corr = train2[cols_t].corr()['target'].abs()
        corr_index = corr[corr > 0.01].drop(labels=['target']).index
        var = train2[cols].var()
        var_index = var[var > 2].index
        columns = var_index.intersection(corr_index)
        idx1 = train2.index; idx2 = test2.index
        train2.reset_index(drop=True,inplace=True)

        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
        #pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])
        pipe1 = Pipeline([('scaler', StandardScaler())])
        data1 = pd.concat([pd.DataFrame(train2[columns]), pd.DataFrame(test2[columns])])
        data1 = pipe1.fit_transform(data1[columns])
        train3 = data1[:train2.shape[0]]
        test3 = data1[train2.shape[0]:]
        train3_gm = model[i].predict_proba(train3)
        test3_gm = model[i].predict_proba(test3)
        train_label = model[i].predict(train3)
        test_label = model[i].predict(test3)
        clf = QuadraticDiscriminantAnalysis(reg_param=0.225)
        clf.fit(np.vstack((train3, test3)), np.concatenate((train_label, test_label)))
        train3_qda = clf.predict_proba(train3)
        test3_qda = clf.predict_proba(test3)
        train3 = np.hstack((train3_gm, train3_qda))
        test3 = np.hstack((test3_gm, test3_qda))
        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
        for train_index, test_index in skf.split(train3, train2['target']):
            abs_train_error = np.abs(train2.loc[train_index]['target'].values - oof[idx1[train_index]])
            abs_err_threshold = np.percentile(abs_train_error[abs_train_error > 0.5], 10)
            mask = (abs_train_error < 0.5) | (abs_train_error > abs_err_threshold)
            train_index = train_index[mask]
            x_train = train3[train_index,:]
            y_train = train2.loc[train_index]['target'].values
            y_train[abs_train_error[mask] > abs_err_threshold] = 1 - y_train[abs_train_error[mask] > abs_err_threshold]
            x_val = train3[test_index,:]
            clf = QuadraticDiscriminantAnalysis(reg_param=0.075)
            clf.fit(x_train,y_train)
            oof2[idx1[test_index]] = clf.predict_proba(x_val)[:,1]
            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
    # PRINT CV AUC
    oof = oof2.copy()
    auc = roc_auc_score(train['target'],oof2)
    print('CV scores = ',round(auc,5))


# In[ ]:


sub = pd.read_csv('../input/instant-gratification/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)

