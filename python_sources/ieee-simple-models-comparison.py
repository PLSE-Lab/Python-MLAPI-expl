#!/usr/bin/env python
# coding: utf-8

# ## Motivation
# This kernel is dedicated to show simple models comparison, inspired by Pavel Pleskov [pipeline shared during Kaggle Days at Dubai](https://gitlab.com/ppleskov/kaggle-days-dubai). 
# 
# In this kenel I am comparing standard algos used for classification task in tabular data. 
# 
# Evaluation metric is [ROC-AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc). 
# 
# <left><img src='https://images.martechadvisor.com/images/uploads/content_images/frauddetectio_5b60873e86283.jpg'>`

# In[ ]:


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from catboost import CatBoostClassifier, Pool
import random 

import os
from os import listdir
from tqdm import tqdm
from os.path import isfile

import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD

from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import time
import datetime

#import shap
# load JS visualization code to notebook
#shap.initjs()

import warnings
warnings.filterwarnings("ignore")

print(os.listdir("../input"))
print()

print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("sklearn:", sklearn.__version__)
print()
print("lightgbm:", lgb.__version__)
print("xgboost:", xgb.__version__)
print("catboost:", cb.__version__)


# ### Loading data

# In[ ]:


train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')


# Let's merge data as in [starter kernel](https://www.kaggle.com/inversion/ieee-simple-xgboost):

# In[ ]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(f'Shape of train set: {train.shape}')
print(f'Shape of test set: {test.shape}')


# Based on 590k entries in train we should predict 506k for test set.
# 
# Let's take a look on target distribution:

# In[ ]:


sns.countplot(train['isFraud']) #Imbalanced Dataset
plt.title('Target distribution');


# In[ ]:


print(f'Number of fraud samples in train: {len(np.where(train["isFraud"]==1)[0])}')
print(f'Percent of fraud samples in train: {round(100.0*len(np.where(train["isFraud"]==1)[0])/len(train["isFraud"]),2)}')


# ### Now, let's take a 10 % sample of the dataset to speed up all calculations

# In[ ]:


train = train.sample(frac=0.1, random_state=42) # comment if you want to run on entire set (takes longer time)
train.reset_index(drop=True, inplace=True)


# In[ ]:


y = train.isFraud.values

train = train.drop('isFraud', axis=1)
test = test.copy()
train = train.fillna(-1) #nan substitution could be done in a better way
test = test.fillna(-1) 
del train_transaction, train_identity, test_transaction, test_identity


# In[ ]:


# Label Encoding
for f in train.columns:
    if train[f].dtype=='object' or test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))  


# In[ ]:


cols = list(train.columns)
len(cols)


# ### Standard scaler preprocessing

# In[ ]:


scaler = StandardScaler() #MinMaxScaler StandardScaler RobustScaler

train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])


# In[ ]:


N = 50

svd = TruncatedSVD(n_components=N, random_state=42)
X = svd.fit_transform(train[cols], y)  
svd.explained_variance_ratio_.sum()


# In[ ]:


df = pd.DataFrame()
df["target"] = y

for i in range(50):
    df[i] = X[:,i]
    
df.tail()


# ## Logistic Regression

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\narch = "reg"\n\ntrain[arch] = 0\n\nfor i, (train_index, valid_index) in enumerate(skf.split(X, y)):\n    \n    X_train = X[train_index]\n    X_valid = X[valid_index]\n\n    y_train = y[train_index]\n    y_valid = y[valid_index]\n    \n    reg = LogisticRegression(C=1,\n                             solver="newton-cg", \n                             penalty="l2", \n                             n_jobs=-1, \n                             max_iter=100).fit(X_train, y_train) \n    \n    y_pred = reg.predict_proba(X_valid)[:,1]\n    train.loc[valid_index, arch] = y_pred\n    print(i, "ROC AUC:", round(roc_auc_score(y_valid, y_pred), 5))\n\nprint()\nprint("OOF ROC AUC:", round(roc_auc_score(y, train[arch]), 5))\nprint()')


# ## Random Forest

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\narch = "rfc"\n\ntrain[arch] = 0\ntest[arch] = 0\n\nfor i, (train_index, valid_index) in enumerate(skf.split(X, y)):\n    \n    X_train = X[train_index]\n    X_valid = X[valid_index]\n\n    y_train = y[train_index]\n    y_valid = y[valid_index]\n    \n    rfc = RandomForestClassifier(n_estimators=100,\n                                 criterion=\'gini\',\n                                 n_jobs=-1).fit(X_train, y_train) \n    \n    y_pred = rfc.predict_proba(X_valid)[:,1]\n    train.loc[valid_index, arch] = y_pred\n    print(i, "ROC AUC:", round(roc_auc_score(y_valid, y_pred), 5))\n\nprint()\nprint("OOF ROC AUC:", round(roc_auc_score(y, train[arch]), 5))\nprint()')


# ## LGBM

# In[ ]:


get_ipython().run_cell_magic('time', '', '\narch = "lgb"\n\ntrain[arch] = 0\n\nrounds = 10000\nearly_stop_rounds = 300\n\nparams = {\'objective\': \'binary\',\n          \'boosting_type\': \'gbrt\',\n          \'metric\': \'auc\',\n          \'seed\': 42,\n          \'max_depth\': -1,\n          \'verbose\': -1,\n          \'n_jobs\': -1}\n\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n\nfor i, (train_index, valid_index) in enumerate(skf.split(X, y)):\n    \n    X_train = X[train_index]\n    X_valid = X[valid_index]\n\n    y_train = y[train_index]\n    y_valid = y[valid_index]\n\n    d_train = lgb.Dataset(X_train, y_train)\n    d_valid = lgb.Dataset(X_valid, y_valid)    \n\n    model = lgb.train(params,\n                      d_train,\n                      num_boost_round=rounds,\n                      valid_sets=[d_train, d_valid],\n                      valid_names=[\'train\',\'valid\'],\n                      early_stopping_rounds=early_stop_rounds,\n                      verbose_eval=0) \n\n\n    y_pred = model.predict(X_valid)\n    train.loc[valid_index, arch] = y_pred\n    auc = roc_auc_score(y_valid, y_pred)\n    print(i, "ROC AUC:", round(auc, 5))\n\nprint()\nprint("OOF ROC AUC:", round(roc_auc_score(y, train[arch]), 5))\nprint()')


# ## Catboost

# In[ ]:


get_ipython().run_cell_magic('time', '', '\narch = "cat"\n\ntrain[arch] = 0\n\nrounds = 10000\nearly_stop_rounds = 100\n\nparams = {\'task_type\': \'CPU\', #GPU\n          \'iterations\': rounds,\n          \'loss_function\': \'Logloss\',\n          \'eval_metric\':\'AUC\',\n          \'random_seed\': 42,\n          \'learning_rate\': 0.5,\n          \'depth\': 2}\n\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n\nfor i, (train_index, valid_index) in enumerate(skf.split(X, y)):\n    \n    X_train = X[train_index]\n    X_valid = X[valid_index]\n\n    y_train = y[train_index]\n    y_valid = y[valid_index]\n    \n    trn_data = Pool(X_train, y_train)\n    val_data = Pool(X_valid, y_valid)\n    \n    clf = CatBoostClassifier(**params)\n    clf.fit(trn_data,\n            eval_set=val_data,\n            use_best_model=True,\n            early_stopping_rounds=early_stop_rounds,\n            verbose=0)\n    \n    y_pred = clf.predict_proba(X_valid)[:, 1]\n    train.loc[valid_index, arch] = y_pred\n    auc = roc_auc_score(y_valid, y_pred)\n    print(i, "ROC AUC:", round(auc, 5))\n\nprint()\nprint("OOF ROC AUC:", round(roc_auc_score(y, train[arch]), 5))\nprint()')


# ## NN 
# 
# I also checked FastAI implementation described in Pavel's pipeline, but it gives much less score than previous models.

# ## Correlation of the models

# In[ ]:


models = ["cat", "lgb", "rfc", "reg"] #"nn"

for model in models:
    train[model] = train[model].rank()/len(train)

train[models].corr(method="spearman")


# In[ ]:


for arch in models:
    print(arch, round(roc_auc_score(y, train[arch]), 5))


# Maybe now you know which model to tune. 

# ### Blending

# In[ ]:


train["avg"] = train[models].mean(axis=1)
print("avg", round(roc_auc_score(y, train["avg"]), 5))


# In[ ]:


from scipy.stats.mstats import gmean

def power_mean(x, p=1):
    if p==0:
        return gmean(x, axis=1)
    return np.power(np.mean(np.power(x,p), axis=1), 1/p)


# In[ ]:


for power in [0,1,2,4,8]:
    train["avg"] = power_mean(train[models].values, power)
    print(power, round(roc_auc_score(y, train["avg"]), 5))


# ## Stacking
# 
# Let's stack predictions of previos model and learn LR on them. 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\narch = "stack"\n\ntrain[arch] = 0\n\nfor i, (train_index, valid_index) in enumerate(skf.split(X, y)):\n    \n    X_train = train.loc[train_index, models]\n    X_valid = train.loc[valid_index, models]\n\n    y_train = y[train_index]\n    y_valid = y[valid_index]\n    \n    reg = LogisticRegression(C=1,\n                             solver="newton-cg", \n                             penalty="l2", \n                             n_jobs=-1, \n                             max_iter=100).fit(X_train, y_train) \n    \n    y_pred = reg.predict_proba(X_valid)[:,1]\n    train.loc[valid_index, arch] = y_pred\n    print(i, "ROC AUC:", round(roc_auc_score(y_valid, y_pred), 5))\n\nprint()\nprint("OOF ROC AUC:", round(roc_auc_score(y, train[arch]), 5))\nprint()')


# ## What is not covered (yet)
# 
# - SVM
# - KNN
# - FastAI (because of low score)
# - H2O (that approach is already shown in Bojan's kernel: https://www.kaggle.com/tunguz/ieee-with-h2o-automl)

# ### Reference: 
# 
# Most of the code is taken from https://gitlab.com/ppleskov/kaggle-days-dubai as I wrote in the beginning of the kernel.
# 
# But it's applies it for Fraud detection task and gives a hint about models which could be used in that competition.
