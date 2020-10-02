#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Nativos
import os
import sys

#calculo
import numpy as np
import pandas as pd

#modelamiento
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

#warning ignore future
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import os

subfolder = "../input"
print(os.listdir(subfolder))


# ### LOAD DATA

# In[ ]:


set_parameter_csv = {
    'sep': ',',
    'encoding': 'ISO-8859-1',
    'low_memory': False
}

train = pd.read_csv(subfolder + '/train.csv', **set_parameter_csv).round(2)
test = pd.read_csv(subfolder + '/test.csv', **set_parameter_csv).round(2)
train.shape, test.shape


# In[ ]:


train.head(2)


# ### REDUCE SIZE

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def get_memory_usage(data, deep=True):\n    return \'{} MB\'.format(data.memory_usage(deep=deep).sum() / 1024 ** 2)\n\ndef reduce_size_data(df, default=\'\'):\n    print("INITIAL SIZE : DEEP", get_memory_usage(df), "REAL", get_memory_usage(df, deep=False))\n \n    for col in df.select_dtypes(include=[\'int\']).columns:\n        df[col] = pd.to_numeric(arg=df[col], downcast=default or\'integer\')\n    \n    for col in df.select_dtypes(include=[\'float\']).columns:\n        df[col] = pd.to_numeric(arg=df[col], downcast=default or\'float\')\n                \n    print("FINAL SIZE : DEPP", get_memory_usage(df), "REAL", get_memory_usage(df, deep=False))               \n    return df\n\ntrain = reduce_size_data(train)\ntest = reduce_size_data(test)')


# ### NULL ANALYSIS

# In[ ]:


train.isnull().sum().any(), test.isnull().sum().any()


# ### CATEGORY DETECTION

# In[ ]:


for col in train.columns:
    unicos = train[col].unique().shape[0]
    if unicos < 1000:
        print(col, unicos)


# ### PROCESS VARIABLES

# In[ ]:


SEED = 29082013
pd.np.random.seed(SEED)

col_wctm = 'wheezy-copper-turtle-magic'
col_target = 'target'
col_log = 'predict_log'
col_knn = 'predict_knn'
cols = [c for c in train.columns if c not in ['id', col_wctm, col_target]]

kfold_off = StratifiedKFold(
    n_splits=11, 
    shuffle=False, 
    random_state=SEED
)
param_grid_knn = {
    'n_neighbors': [7],
    'p': [2],
    'weights':['distance']
}
param_grid_gauss = {
    'priors': [[0.5, 0.5]],
    'reg_param': [0.3]
}

param_grid_log = {
    'solver': ['sag'],
    'penalty': ['l2'],
    'C': [0.001],
    'tol': [0.0001]
}

model_knn = KNeighborsClassifier()
model_gauss = QuadraticDiscriminantAnalysis()
model_log = LogisticRegression(random_state=SEED)
km = KMeans(n_clusters=5, n_init=5, init='k-means++', random_state=SEED, algorithm='elkan')
pca = PCA(svd_solver='full',n_components='mle')


# ### FUNCTIONS MODELING

# In[ ]:


def apply_pca(X_train, X_test):
    #PCA
    X_train = pd.DataFrame(pca.fit_transform(X_train))
    X_test = pd.DataFrame(pca.transform(X_test)) 
    return X_train, X_test

def apply_km(X_train, X_test):
    col_km = 'cluster_km'
    X_train[col_km] = km.fit_predict(X_train)
    X_test[col_km] = km.predict(X_test)
    return pd.get_dummies(X_train, columns=[col_km]), pd.get_dummies(X_test, columns=[col_km])

def apply_grid(X_train, y_train, X_test, model, param_grid, predict_train=True):
    grid = GridSearchCV(
        model, param_grid, cv=kfold_off, n_jobs=-1, scoring='roc_auc'
    )
    grid.fit(X_train, y_train)
    print(grid.best_score_, end=' / ')
    if predict_train:
        return grid.best_estimator_.predict_proba(X_train)[:,1], grid.best_estimator_.predict_proba(X_test)[:,1]
    else:
        return grid.best_estimator_.predict_proba(X_test)[:,1]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'col_wctm = \'wheezy-copper-turtle-magic\'\ncol_target = \'target\'\ncols = [c for c in train.columns if c not in [\'id\', col_wctm, col_target]]\nresult = []\nscores = 0\nscores2 = 0\n\nfor val in sorted(train[col_wctm].unique()):\n    # Build X_train and y_train\n    X_train = train[train[col_wctm] == val]\n    y_train = X_train[col_target]\n    X_test = test[test[col_wctm] == val]\n    result_test = X_test[[\'id\', col_wctm]]\n    \n    X_train = X_train[cols]\n    X_test = X_test[cols]\n    \n    #PCA\n    X_train, X_test = apply_pca(X_train, X_test)\n    \n    # ADD column prediction log or knn\n    train_log, test_log = apply_grid(X_train, y_train, X_test, model_log, param_grid_log)\n    train_knn, test_knn = apply_grid(X_train, y_train, X_test, model_knn, param_grid_knn)    \n    \n    X_train, X_test = apply_km(X_train, X_test)\n    \n    X_train[col_log] = train_log\n    X_test[col_log] = test_log\n    X_train[col_knn] = train_knn\n    X_test[col_knn] = test_knn\n    \n    # TRAIN    \n    result_test[col_target] = apply_grid(X_train, y_train, X_test, model_gauss, param_grid_gauss, predict_train=False)\n    result.append(result_test[[\'id\', col_target]])\n    print("-"*100)')


# In[ ]:


result = pd.concat(result).sort_index()
result.head(15)


# In[ ]:


"""
def fix_round(val):
    if val > 0.9:
        return 1
    elif val < 0.1:
        return 0
    else:
        return val
    
result[col_target] = result[col_target].apply(lambda _: fix_round(_))
"""


# In[ ]:


result.to_csv('oordered_log_kfold11_knn7_qda_38.csv', index=False)


# In[ ]:





# In[ ]:




