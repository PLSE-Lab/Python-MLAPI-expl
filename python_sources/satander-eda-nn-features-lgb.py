#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# This kernel is dedicated to EDA of Santander Value Prediction Challenge competition as well as feature engineering and some modeling.

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
import lightgbm as lgb
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
kf = KFold(n_splits=10)


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## Data overview

# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# We have an interesting situation here: there are more features than samples in train. And test is 10x times larger than train. In will be difficult to train models adequately. So stacking and blending will be the ways to go...

# In[ ]:


plt.hist(train.target);
plt.title('Target histogram.');


# In[ ]:


plt.hist(np.log1p(train.target));
plt.title('Logarithm transformed target histogram.');


# Target variable is really skewered

# It is difficult to analyze anonymized features especially considering their number, but let's try.
# 
# At first let's see at the number of unique values in columns.

# In[ ]:


unique_values = [len(train[col].unique()) for col in train.columns]


# In[ ]:


pd.Series(unique_values).quantile([0.25, 0.50, 0.75])


# In[ ]:


pd.Series(unique_values).value_counts()


# In[ ]:


train[[col for col in train.columns if len(train[col].unique()) == 1]].head()


# 25% of features have 7 or less unque values - I suppose most of them are categorical. And 256 features have all zero values. I'll drop them.

# In[ ]:


train[[col for col in train.columns if len(train[col].unique()) == 2]].describe()


# In[ ]:


zero_count = []
for col in [col for col in train.columns if len(train[col].unique()) == 2]:
    zero_count.append([i[1] for i in list(train[col].value_counts().items()) if i[0] == 0][0])
    
print('{0} features of 245 having 2 unique values have zeroes in 99% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.99])))


# So most of these features are useless. In fact, let's repeat this analysis for all features.

# In[ ]:


zero_count = []
for col in train.columns[2:]:
    zero_count.append([i[1] for i in list(train[col].value_counts().items()) if i[0] == 0][0])
    
print('{0} features of 4491 have zeroes in 99% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.99])))
print('{0} features of 4491 have zeroes in 98% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.98])))
print('{0} features of 4491 have zeroes in 97% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.97])))
print('{0} features of 4491 have zeroes in 96% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.96])))
print('{0} features of 4491 have zeroes in 95% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.95])))


# Well... this is a problem. Maybe having 95% zeroes isn't than bad, but features with 98-99% zeroes are very likely to be useless.
# I'll drop features where 98% or more samples are zeroes and will try some modelling.

# In[ ]:


cols_to_drop = [col for col in train.columns[2:] if [i[1] for i in list(train[col].value_counts().items()) if i[0] == 0][0] >= 4459 * 0.98]


# In[ ]:


train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)


# ## Modelling

# In[ ]:


X = train.drop(['ID', 'target'], axis=1)
y = train['target']
X_test = test.drop('ID', axis=1)


# ### Ridge regression

# In[ ]:


ridge = Ridge()


# In[ ]:


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)


# In[ ]:


-cross_val_score(ridge, X, y, scoring=rmsle_scorer)


# It isn't surprising that the result isn't great - ridge maybe unable to work with our features. Let's try LGB!

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
params = {'learning_rate': 0.01, 'max_depth': 6, 'boosting': 'gbdt', 'objective': 'regression', 'metric': ['rmse'], 'is_training_metric': True, 'seed': 19, 'num_leaves': 63, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=50, early_stopping_rounds=20)
print('RMSLE on valid data: {0:.4}.'.format(rmsle(y_valid, model.predict(X_valid))))


# This is better. Now let's try log transformation on y.

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, np.log1p(y), test_size=0.20, random_state=42)
params = {'learning_rate': 0.01, 'max_depth': 6, 'boosting': 'gbdt', 'objective': 'regression', 'metric': ['rmse'], 'is_training_metric': True, 'seed': 19, 'num_leaves': 63, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=50, early_stopping_rounds=20)
print('RMSLE on valid data: {0:.4}.'.format(rmsle(np.expm1(y_valid), np.expm1(model.predict(X_valid)))))


# This is better! Now we can try other parameters.

# In[ ]:


params = {'learning_rate': 0.01, 'max_depth': 13, 'boosting': 'gbdt', 'objective': 'regression_l2', 'metric': ['rmse'], 'is_training_metric': True, 'seed': 19, 'num_leaves': 26, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=50, early_stopping_rounds=20)
print('RMSLE on valid data: {0:.4}.'.format(rmsle(np.expm1(y_valid), np.expm1(model.predict(X_valid)))))


# In[ ]:


params = {'learning_rate': 0.01, 'max_depth': 13, 'boosting': 'rf', 'objective': 'regression_l2', 'metric': ['rmse'], 'is_training_metric': True, 'seed': 19, 'num_leaves': 256, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=50, early_stopping_rounds=20)
print('RMSLE on valid data: {0:.4}.'.format(rmsle(np.expm1(y_valid), np.expm1(model.predict(X_valid)))))


# In[ ]:


params = {'learning_rate': 0.01, 'max_depth': 3, 'boosting': 'gbdt', 'objective': 'regression_l2', 'metric': ['rmse'], 'is_training_metric': True, 'seed': 19, 'num_leaves': 8, 'feature_fraction': 0.9, 'bagging_fraction': 0.8, 'bagging_freq': 5}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=50, early_stopping_rounds=20)
print('RMSLE on valid data: {0:.4}.'.format(rmsle(np.expm1(y_valid), np.expm1(model.predict(X_valid)))))


# In[ ]:


params = {'learning_rate': 0.01, 'max_depth': 13, 'boosting': 'gbdt', 'objective': 'regression', 'metric': ['rmse'], 'is_training_metric': True, 'seed': 19, 'num_leaves': 128, 'feature_fraction': 0.9,
          'bagging_fraction': 0.8, 'bagging_freq': 5, 'num_threads': 16}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=50, early_stopping_rounds=20)
print('RMSLE on valid data: {0:.4}.'.format(rmsle(np.expm1(y_valid), np.expm1(model.predict(X_valid)))))


# In[ ]:


params = {'learning_rate': 0.02, 'max_depth': 13, 'boosting': 'gbdt', 'objective': 'regression', 'metric': 'rmse', 'is_training_metric': True, 'num_leaves': 12**2, 'feature_fraction': 0.9,
          'bagging_fraction': 0.8, 'bagging_freq': 5,  'num_threads': 16}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=50, early_stopping_rounds=20)
print('RMSLE on valid data: {0:.4}.'.format(rmsle(np.expm1(y_valid), np.expm1(model.predict(X_valid)))))


# ## NN features
# And now let's try to add some new features. I'll use NearestNeighbors model and find statistics on similar rows.

# In[ ]:


df = pd.concat([X, X_test])
X.shape, df.shape


# In[ ]:


neigh = NearestNeighbors(5, n_jobs=-1)
neigh.fit(df)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'dists, _ = neigh.kneighbors(X, n_neighbors=3)')


# In[ ]:


mean_dist = dists.mean(axis=1)
max_dist = dists.max(axis=1)
min_dist = dists.min(axis=1)


# In[ ]:


X_ = np.hstack((X, mean_dist.reshape(-1, 1), max_dist.reshape(-1, 1), min_dist.reshape(-1, 1)))


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X_, np.log1p(y), test_size=0.20, random_state=42)
params = {'learning_rate': 0.02, 'max_depth': 13, 'boosting': 'gbdt', 'objective': 'regression', 'metric': 'rmse', 'is_training_metric': True, 'num_leaves': 12**2, 'feature_fraction': 0.9,
          'bagging_fraction': 0.8, 'bagging_freq': 5,  'num_threads': 16}
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=50, early_stopping_rounds=20)
print('RMSLE on valid data: {0:.4}.'.format(rmsle(np.expm1(y_valid), np.expm1(model.predict(X_valid)))))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_dists, _ = neigh.kneighbors(X_test, n_neighbors=3)')


# In[ ]:


test_mean_dist = test_dists.mean(axis=1)
test_max_dist = test_dists.max(axis=1)
test_min_dist = test_dists.min(axis=1)
X_test_ = np.hstack((X_test, test_mean_dist.reshape(-1, 1), test_max_dist.reshape(-1, 1), test_min_dist.reshape(-1, 1)))


# In[ ]:


1


# In[ ]:


get_ipython().run_cell_magic('time', '', "prediction = np.zeros((test.shape[0], 1))\nscore = []\nfor train_i, test_i in kf.split(X_):\n    print('Fold')\n    X_train = X_[train_i]\n    y_train = np.log1p(y)[train_i]\n    X_valid = X_[test_i]\n    y_valid = np.log1p(y)[test_i]\n    model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, lgb.Dataset(X_valid, label=y_valid), verbose_eval=1000, early_stopping_rounds=100)\n    pred = model.predict(X_test_).reshape(-1, 1)\n    prediction += np.expm1(pred)\n    score.append(model.best_score['valid_0']['rmse'])\nprint('Mean score: {:.6}. Std score: {:.6}'.format(np.mean(score), np.std(score)))")


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = prediction / 10
sub.to_csv('lgb.csv', index=False)

