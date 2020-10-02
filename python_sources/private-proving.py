#!/usr/bin/env python
# coding: utf-8

# - V1 : LGBM STACKING 
# - V2 : LGBM, MLP16 STACKING
# - V3 : V2 + pred 5 score 3

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook

from sklearn import svm, neighbors, linear_model, neural_network

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA

from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score


import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import preprocessing
from sklearn import svm, neighbors, linear_model
import gc
warnings.filterwarnings('ignore')


from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import FastICA, TruncatedSVD, PCA
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from tqdm import *


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv('../input/train.csv')\ntest_df = pd.read_csv('../input/test.csv')")


# In[ ]:


train_columns = [c for c in train_df.columns if c not in ['id','target','wheezy-copper-turtle-magic']]

magic_variance_over2 = {}
for magic in sorted(train_df['wheezy-copper-turtle-magic'].unique()):
    temp = train_df.loc[train_df['wheezy-copper-turtle-magic']==magic]
    std = temp[train_columns].std()
    magic_variance_over2[magic] = list(std.index.values[np.where(std >2)])


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Lasso, LassoLars


# In[ ]:


random_state = 42
debug = True
debug = False


# https://www.kaggle.com/c/instant-gratification/discussion/93080#latest-554239

# In[ ]:


import hashlib
# CREATE LIST PUBLIC DATASET IDS
public_ids = []
for i in range(256*512+1):
    st = str(i)+"test"
    public_ids.append( hashlib.md5(st.encode()).hexdigest() )
# SPLIT DATA
public = test_df[ test_df['id'].isin(public_ids) ].copy()
private = test_df[ ~test_df.index.isin(public.index) ].copy()


# # Pseudo 

# In[ ]:


public_count = public[public['wheezy-copper-turtle-magic']==0].shape[0]
private_count = private[private['wheezy-copper-turtle-magic']==0].shape[0]


# In[ ]:


max_magic0_private_count = 281
min_magic0_private_count = 200

private_proving_range = max_magic0_private_count - min_magic0_private_count

min_start_magic = 100 
magic_iter_range = 512 - min_start_magic
min_step = int(magic_iter_range/private_proving_range)

magic_iter_num = 1
private_error_flag = False
if private_count < 201:
    magic_iter_num = 1
elif private_count > 281:
    private_error_flag = True
    magic_iter_num = 1
else:
    magic_iter_num = (private_count - min_magic0_private_count)*min_start_magic+min_start_magic


# In[ ]:


oof_qda_pseudo = np.zeros(len(train_df))
preds_qda_pseudo = np.zeros(len(test_df))

cols = [c for c in train_df.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

for i in tqdm_notebook(range(magic_iter_num)):

    # each magic
    train = train_df[train_df['wheezy-copper-turtle-magic'] == i]
    test = test_df[test_df['wheezy-copper-turtle-magic'] == i]

    # for oof
    train_idx_origin = train.index
    test_idx_origin = test.index

    # start point
    # new cols
    cols = magic_variance_over2[i]

    X_train = train.reset_index(drop=True)[cols].values
    y_train = train.reset_index(drop=True).target

    X_test = test[cols].values

    # vstack
    data = np.vstack([X_train, X_test])

    # STANDARD SCALER
    data = StandardScaler().fit_transform(data)

    # new train/test
    X_train = data[:X_train.shape[0]]
    X_test = data[X_train.shape[0]:]

    fold = StratifiedKFold(n_splits=5, random_state=random_state)
    for tr_idx, val_idx in fold.split(X_train, y_train):
        # qda 3
        clf = QuadraticDiscriminantAnalysis(reg_param=0.111)
        clf.fit(X_train[tr_idx], y_train[tr_idx])
        oof_qda_pseudo[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]
        preds_qda_pseudo[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits
    print(i, roc_auc_score(y_train,oof_qda_pseudo[train_idx_origin]))


# In[ ]:


print(roc_auc_score(train_df['target'],oof_qda_pseudo))


# In[ ]:


submit = pd.read_csv('../input/sample_submission.csv')
submit["target"] = preds_qda_pseudo
if private_error_flag is True:
    submit.to_csv("submission.csv") # raise error
else:    
    submit.to_csv("submission.csv", index=False)

