#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.isnull().sum().any()


# In[ ]:


target = train['target']
f = train.drop(['target', 'ID_code'] , axis = 1)
testf = test.drop(['ID_code'], axis = 1 )


# In[ ]:


from sklearn.decomposition import PCA, FastICA
pca = PCA(n_components=12, random_state=420)
pca2_results_train = pca.fit_transform(f)
pca2_results_test = pca.transform(testf)


# In[ ]:


ica = FastICA(n_components=12, random_state=420)
ica2_results_train = ica.fit_transform(f)
ica2_results_test = ica.transform(testf)


# In[ ]:


trainNN = pd.DataFrame()
testNN = pd.DataFrame()


# In[ ]:


for i in range(1, 13):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]

    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]


# In[ ]:


from sklearn.decomposition import TruncatedSVD


# In[ ]:


tsvd = TruncatedSVD(n_components=15, random_state=420)
tsvd_results_train = tsvd.fit_transform(f)
tsvd_results_test = tsvd.transform(testf)


# In[ ]:


from sklearn.random_projection import GaussianRandomProjection
grp = GaussianRandomProjection(n_components=15, eps=0.2, random_state=420)
grp_results_train = grp.fit_transform(f)
grp_results_test = grp.transform(testf)


# In[ ]:


from sklearn.random_projection import SparseRandomProjection
srp = SparseRandomProjection(n_components=15, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(f)
srp_results_test = srp.transform(testf)


# In[ ]:


for i in range(1, 16):
    trainNN['tsvd_' + str(i)] = tsvd_results_train[:,i-1]
    testNN['tsvd_' + str(i)] = tsvd_results_test[:, i-1]

    trainNN['grp_' + str(i)] = grp_results_train[:,i-1]
    testNN['grp_' + str(i)] = grp_results_test[:, i-1]

    trainNN['srp_' + str(i)] = srp_results_train[:,i-1]
    testNN['srp_' + str(i)] = srp_results_test[:, i-1]


# In[ ]:


import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 520,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.98,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


# In[ ]:


dtrain = xgb.DMatrix(f, target)
dtest = xgb.DMatrix(testf)


# In[ ]:


num_boost_rounds = 300


# In[ ]:


model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)


# In[ ]:


y_pred = model.predict(dtest)


# In[ ]:


trainNN.shape


# In[ ]:


import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_paramst = {
    'n_trees': 300,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


# In[ ]:


dttrain = xgb.DMatrix(trainNN, target)
dttest = xgb.DMatrix(testNN)


# In[ ]:


num_boost_roundst = 800


# In[ ]:


model2 = xgb.train(dict(xgb_paramst, silent=0), dttrain, num_boost_round=num_boost_rounds)


# In[ ]:


t_pred = model2.predict(dttest)


# In[ ]:


sub = pd.DataFrame()
sub['ID_code'] = test['ID_code']
sub['target'] = y_pred*0.65 + t_pred*0.35
sub.to_csv('stacked-models.csv', index=False)


# In[ ]:


sub.head()

