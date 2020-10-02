#!/usr/bin/env python
# coding: utf-8

# ## LGBM is Powerful!
# 
# I want to prove LGBM is better than Logistic Regression. 
# 
# Let's try :)

# ## Import Libarary & Read CSV

# In[ ]:


import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import seaborn as sns
import numpy as np 
import pandas as pd
import os, gc
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
sns.set()


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')\ntest_df = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')")


# In[ ]:


train_df.shape


# In[ ]:


target = train_df['target']
train_id = train_df['id']
test_id = test_df['id']
train_df.drop(['target', 'id'], axis=1, inplace=True)
test_df.drop('id', axis=1, inplace=True)


# In[ ]:


print(train_df.shape)
print(test_df.shape)


# In[ ]:


train_df.head()


# ## Feature Engineering (Target Encoding)

# and remaining loooooong features : target encoding

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntraintest = pd.concat([train_df, test_df])\ndummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)\ntrain = dummies.iloc[:train_df.shape[0], :]\ntest = dummies.iloc[train_df.shape[0]:, :]\ntrain = train.sparse.to_coo().tocsr()\ntest = test.sparse.to_coo().tocsr()')


# In[ ]:


train = train.astype('float32')
test = test.astype('float32')


# ## LightGBM model

# This is my first single LGBM Model (public leaderboard score low)

# In[ ]:


# %%time 
# X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=97)

# param = {   
#     'boost': 'gbdt',
#     'learning_rate': 0.005,
#     'feature_fraction':0.3,
#     'bagging_freq':1,
#     'max_depth': -1,
#     'num_leaves':18,
#     'lambda_l2': 3,
#     'lambda_l1': 3,
#     'metric':{'auc'},
#     'tree_learner': 'serial',
#     'objective': 'binary',
#     'verbosity': 1,
#     'seed': 97,
#     'feature_fraction_seed': 97,
#     'bagging_seed': 97,
#     'drop_seed': 97,
#     'data_random_seed': 97,
# }


# evals_result = {}
# predictions = np.zeros(test.shape[0])

# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_valid = lgb.Dataset(X_test, y_test)

# num_round = 20000
# clf = lgb.train(param, lgb_train, num_round, valid_sets = [lgb_train, lgb_valid],
#       verbose_eval=100, early_stopping_rounds = 1000, evals_result = evals_result)

# ## Prediction
# predictions = clf.predict(test, num_iteration=clf.best_iteration)


# ## LGBM with CV

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# CV function original : @Peter Hurford : Why Not Logistic Regression? https://www.kaggle.com/peterhurford/why-not-logistic-regression\nfrom sklearn.model_selection import KFold\nfrom sklearn.metrics import roc_auc_score as auc\n\ndef run_cv_model(train, test, target, model_fn, params={}, label='model'):\n    kf = KFold(n_splits=5)\n    fold_splits = kf.split(train, target)\n\n    cv_scores = []\n    pred_full_test = 0\n    pred_train = np.zeros((train.shape[0]))\n    i = 1\n    for dev_index, val_index in fold_splits:\n        print('Started {} fold {}/5'.format(label, i))\n        dev_X, val_X = train[dev_index], train[val_index]\n        dev_y, val_y = target[dev_index], target[val_index]\n        \n        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params)\n        \n        pred_full_test = pred_full_test + pred_test_y\n        pred_train[val_index] = pred_val_y\n        \n        cv_score = auc(val_y, pred_val_y)\n        cv_scores.append(cv_score)\n        print(label + ' cv score {}: {}\\n'.format(i, cv_score))\n        i += 1\n        \n    print('{} cv scores : {}'.format(label, cv_scores))\n    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))\n    print('{} cv std score : {}'.format(label, np.std(cv_scores)))\n    pred_full_test = pred_full_test / 5.0\n    results = {'label': label, 'train': pred_train, 'test': pred_full_test, 'cv': cv_scores}\n    return results\n\n\ndef runLGBM(X_train, y_train, X_val, y_val, X_test, params):\n    predictions = np.zeros(test.shape[0])\n    lgb_train, lgb_valid = lgb.Dataset(X_train, y_train), lgb.Dataset(X_val, y_val)\n    num_round = 5000\n    clf = lgb.train(params, lgb_train, num_round, valid_sets = [lgb_train, lgb_valid], verbose_eval=1000, early_stopping_rounds = 1000)\n    pred_val_y = clf.predict(X_val, num_iteration=clf.best_iteration)\n    pred_test_y = clf.predict(X_test, num_iteration=clf.best_iteration)\n    return pred_val_y, pred_test_y\n\nparams = {   \n    'boost': 'gbdt',\n    'learning_rate': 0.005,\n    'feature_fraction':0.3,\n    'bagging_freq':1,\n    'max_depth': 1<<5,\n    'num_leaves':18,\n    'lambda_l2': 0.9,\n    'lambda_l1': 0.9,\n    'metric':{'auc'},\n    'tree_learner': 'serial',\n    'objective': 'binary',\n    'verbosity': 1,\n    'seed': 97,\n    'feature_fraction_seed': 97,\n    'bagging_seed': 97,\n    'drop_seed': 97,\n    'data_random_seed': 97,\n}\n\nresults = run_cv_model(train, test, target, runLGBM, params, 'LGBM')")


# ## Result

# In[ ]:


sub_df = pd.DataFrame({'id': test_id, 'target' : results['test']})

sub_df.to_csv("lightgbm_onehotencoding_cv.csv", index=False)
sub_df.head()


# In[ ]:




