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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import os
from sklearn.preprocessing import StandardScaler, scale
import lightgbm as lgb

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
submit_data = pd.read_csv('../input/sample_submission.csv')
features = [c for c in train_data.columns if c not in ['ID_code', 'target']]
target = train_data['target']
print ("Data is ready!")


# In[ ]:


tar = target


# In[ ]:


train_data = train_data.drop(["ID_code"], axis=1)
test_data = test_data.drop(["ID_code"], axis=1)


# In[ ]:


target0 = train_data[target==0]
target1 = train_data[target==1]


# In[ ]:


target0.shape


# In[ ]:


for i in range(0,199):
    ch = 'var_' + str(i)
    target0[ch] = np.random.permutation(target0[ch])
for i in range(0,199):
    ch = 'var_' + str(i)
    target1[ch] = np.random.permutation(target1[ch])


# In[ ]:


target0 = target0.append(target1)


# In[ ]:


train_data = train_data.append(target0)


# In[ ]:


train_data = train_data.sample(frac=1)


# In[ ]:


target = train_data['target']


# In[ ]:


target.shape


# In[ ]:


train_data = train_data.drop(["target"], axis=1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_data = pd.DataFrame(scale(train_data.values), columns=train_data.columns, index=train_data.index)\ntest_data = pd.DataFrame(scale(test_data.values), columns=test_data.columns, index=test_data.index)')


# In[ ]:


params = {}
params['bagging_freq'] = 5 #reducing it as smaller freq & frac reduce overfitting 
params['bagging_fraction'] = 0.0331
params['random_state'] = 42
params['learning_rate'] = 0.0123
params['boost_from_average'] = False
params['boosting_type'] = 'gbdt'
params['feature_fraction'] = 0.045
params['objective'] = 'binary'
params['metric'] = 'auc'
params['min_data_in_leaf'] = 80
params['num_leaves'] = 13
params['num_threads'] = 8
params['tree_learner'] = 'serial'
params['max_depth'] = -1
params['min_sum_hessian_in_leaf'] = 10.0
params['verbosity'] =  1
params['bagging_seed'] = 42
params['seed'] = 42


# In[ ]:


num_folds = 10
features = [c for c in test_data.columns if c not in ['ID_code', 'target']]
#print(features)
folds = StratifiedKFold(n_splits=num_folds,shuffle=True, random_state=42)
oof = np.zeros(len(train_data))
getVal = np.zeros(len(train_data))
predictions = np.zeros(len(tar))
print(predictions.shape)
feature_importance_df = pd.DataFrame()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_data.values, target.values)):\n    \n    X_train, y_train = train_data.iloc[trn_idx][features], target.iloc[trn_idx]\n    X_valid, y_valid = train_data.iloc[val_idx][features], target.iloc[val_idx]\n    \n    X_tr, y_tr = X_train.values, y_train.values\n    X_tr = pd.DataFrame(X_tr)\n    \n    print("Fold idx:{}".format(fold_ + 1))\n    trn_data = lgb.Dataset(X_tr, label=y_tr)\n    val_data = lgb.Dataset(X_valid, label=y_valid)\n    \n    clf = lgb.train(params, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)\n   \n    predictions += clf.predict(test_data[features], num_iteration=clf.best_iteration) / folds.n_splits')


# In[ ]:


submit_data['target'] = pd.DataFrame(predictions)
submit_data.to_csv("LGBM.csv", index=False)

