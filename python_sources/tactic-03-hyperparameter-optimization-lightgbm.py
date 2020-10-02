#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to optimize the Extra-trees model.
# 
# First, all [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier) parameters are analysed separately.
# 
# Then, a grid search is carried out.
# This is a search through all the combinations of parameters,
# which optimize the internal score in the train set.
# 
# The results are collected at [Tactic 03. Hyperparameter optimization](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization).

# In[ ]:


import pip._internal as pip
pip.main(['install', '--upgrade', 'numpy==1.17.2'])
import numpy as np

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

from lwoku import RANDOM_STATE, N_JOBS, VERBOSE, get_prediction
from grid_search_utils import plot_grid_search, table_grid_search

import pickle


# In[ ]:


VERBOSE=1


# # Prepare data

# In[ ]:


# Read training and test files
X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')
X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')

# Define the dependent variable
y_train = X_train['Cover_Type'].copy()

# Define a training set
X_train = X_train.drop(['Cover_Type'], axis='columns')


# # Search over parameters

# In[ ]:


lg_clf = LGBMClassifier(verbosity=VERBOSE,
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS)


# # boosting_type
# ##### : string, optional (default='gbdt')
# 
# - 'gbdt', traditional Gradient Boosting Decision Tree.
# - 'dart', Dropouts meet Multiple Additive Regression Trees.
# - 'goss', Gradient-based One-Side Sampling.
# - 'rf', Random Forest.

# In[ ]:


parameters = {
    'boosting_type': ['gbdt', 'dart', 'goss'] # 'rf' fails
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# The best boosting type is `gbdt`,
# followed by `goss`.
# `dart` is the lesser scored type and with greatest fit time.

# # num_leaves
# ##### : int, optional (default=31)
# 
# Maximum tree leaves for base learners.

# In[ ]:


parameters = {
    'num_leaves': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# More number of leaves, greater the score, up to some point.
# 144 leaves has the greatest score.
# 

# # max_depth
# ##### : int, optional (default=-1)
# 
# Maximum tree depth for base learners, <=0 means no limit.

# In[ ]:


parameters = {
    'max_depth': [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# More `max_depth`, greater the score, up to some point.
# 21 hast the greatest score.

# # learning_rate
# ##### : float, optional (default=0.1)
# 
# Boosting learning rate.
# 
# You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
# in training using ``reset_parameter`` callback.
# 
# Note, that this will ignore the ``learning_rate`` argument in training.

# In[ ]:


parameters = {
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The score increases with the learning rate up to some point.
# From 0.5 begin to decay.
# `learning_rate` 0.5 has the greatest score.

# # n_estimators
# ##### : int, optional (default=100)
# 
# Number of boosted trees to fit.

# In[ ]:


parameters = {
    'n_estimators': [20, 50, 100, 200, 500, 1000, 1500, 1900, 2000, 2100, 2500]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The score and also the fit and score times increase with the number of estimators.
# The greatest score is reached at 2000 estimators.

# # subsample_for_bin
# ##### : int, optional (default=200000)
# 
# Number of samples for constructing bins.

# In[ ]:


parameters = {
    'subsample_for_bin': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The subsample optimum is reached at 987 samples.

# # objective
# ##### : string, callable or None, optional (default=None)
# 
# Specify the learning task and the corresponding learning objective or
# a custom objective function to be used (see note below).
# 
# Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.

# In[ ]:


parameters = {
    'objective': ['regression', 'binary', 'multiclass', 'lambdarank']
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# All the objectives has the same score.

# # class_weight
# ##### : dict, 'balanced' or None, optional (default=None)
# 
# Weights associated with classes in the form ``{class_label: weight}``.
# 
# Use this parameter only for multi-class classification task;
# for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
# Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
# 
# You may want to consider performing probability calibration
# (https://scikit-learn.org/stable/modules/calibration.html) of your model.
# 
# The 'balanced' mode uses the values of y to automatically adjust weights
# inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
# 
# If None, all classes are supposed to have weight one.
# 
# Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
# if ``sample_weight`` is specified.

# In[ ]:


parameters = {
    'class_weight': ['balanced', None, weight]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
table_grid_search(clf, all_ranks=True)


# As all the categories has the same number of samples,
# `balanced` and `None` options has the same result.

# # min_split_gain
# ##### : float, optional (default=0.)
# 
# Minimum loss reduction required to make a further partition on a leaf node of the tree.

# In[ ]:


parameters = {
    'min_split_gain': [x / 10 for x in range(0, 11)] 
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The default value has the greatest score.

# # min_child_weight
# ##### : float, optional (default=1e-3)
# 
# Minimum sum of instance weight (hessian) needed in a child (leaf).

# In[ ]:


parameters = {
    'min_child_weight': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# All child weight scores the same.
# 
# But someones have less score and fit time.
# 
# `1e-15` has the least score and fit time.

# # min_child_samples
# ##### : int, optional (default=20)
# 
# Minimum number of data needed in a child (leaf).

# In[ ]:


parameters = {
    'min_child_samples': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# 5 child samples has a greater score than the default value.

# # subsample
# ##### : float, optional (default=1.)
# 
# Subsample ratio of the training instance.

# In[ ]:


parameters = {
    'subsample': [x / 10 for x in range(1, 11)]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Subsample seems to have no influence over the score.

# # subsample_freq
# ##### : int, optional (default=0)
# 
# Frequence of subsample, <=0 means no enable.

# In[ ]:


parameters = {
    'subsample_freq': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Subsample seems to have no influence over the score.

# # colsample_bytree
# ##### : float, optional (default=1.)
# 
# Subsample ratio of columns when constructing each tree.

# In[ ]:


parameters = {
    'colsample_bytree': [x / 10 for x in range(1, 11)]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# `colsample_bytree` increases the score from 0 to 0.7,
# which is a maximum, and then decays a litte.

# # reg_alpha
# ##### : float, optional (default=0.)
# 
# L1 regularization term on weights.

# In[ ]:


parameters = {
    'reg_alpha': [x / 10 for x in range(0, 11)]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# `reg_alpha` as a descendend trend from 0 to 1,
# but some chaotic.
# 
# It has a maximum at 0.7.

# # reg_lambda
# ##### : float, optional (default=0.)
# 
# L2 regularization term on weights.

# In[ ]:


parameters = {
    'reg_lambda': [x / 10 for x in range(0, 11)]
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# `reg_lambda` as a chaotic behaviour. It has a maximum at 0.4.

# # silent
# ##### : bool, optional (default=True)
# 
# Whether to print messages while running boosting.

# **Note**: Not evaluated

# # importance_type
# ##### : string, optional (default='split')
# 
# The type of feature importance to be filled into ``feature_importances_``.
# If 'split', result contains numbers of times the feature is used in a model.
# If 'gain', result contains total gains of splits which use the feature.

# In[ ]:


parameters = {
    'importance_type': ['split', 'gain']
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Both importance types have the same score.

# # Exhaustive search

# ## First iteration

# In[ ]:


lg_clf = LGBMClassifier(verbosity=VERBOSE,
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS)
lg_clf.min_child_weight = 1e-15
lg_clf.boosting_type = 'gbdt'
lg_clf.num_leaves = 144
lg_clf.max_depth = 21
lg_clf.learning_rate = 0.6
lg_clf.n_estimators = 2000
lg_clf.subsample_for_bin = 987
lg_clf.min_child_samples = 5
lg_clf.colsample_bytree = 0.7
lg_clf.reg_alpha = 0.7
lg_clf.reg_lambda = 0.4
parameters = {
#     'boosting_type': ['gbdt', 'goss'],
#     'num_leaves': [34, 50, 55, 60, 89],
#     'max_depth': [13, 21, 34],
#     'learning_rate': [0.5, 0.6, 0.7],
#     'n_estimators': [1900, 2000, 2100],
#     'subsample_for_bin': [610, 987, 1597],
#     'min_child_samples': [3, 5, 8, 20],
#     'colsample_bytree': [0.85, 0.9, 0.95],
#     'reg_alpha': [0.0, 0.7],
#     'reg_lambda': [0.0, 0.4]
}
# clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
# clf.fit(X_train, y_train)
# plot_grid_search(clf)
# table_grid_search(clf)


# The sum of the improvements of parameters is not the improvements when joining all parameters together.
# 
# This score is below some separate improvement.
# 
# Then, a manual sensibility analysis is done for each of fixed parameters to retune it.

# ## Second iteration

# In[ ]:


lg_clf = LGBMClassifier(verbosity=VERBOSE,
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS)
lg_clf.min_child_weight = 1e-15
lg_clf.boosting_type = 'gbdt'
lg_clf.num_leaves = 55
lg_clf.max_depth = 21
lg_clf.learning_rate = 0.6
lg_clf.n_estimators = 2100
lg_clf.subsample_for_bin = 987
lg_clf.min_child_samples = 5
lg_clf.colsample_bytree = 0.9
lg_clf.reg_alpha = 0.0
lg_clf.reg_lambda = 0.0
parameters = {
#     'boosting_type': ['gbdt', 'goss'],
#     'num_leaves': [34, 50, 55, 60, 89],
#     'max_depth': [13, 21, 34],
#     'learning_rate': [0.5, 0.6, 0.7],
#     'n_estimators': [1900, 2000, 2100],
#     'subsample_for_bin': [610, 987, 1597],
#     'min_child_samples': [3, 5, 8, 20],
#     'colsample_bytree': [0.85, 0.9, 0.95],
#     'reg_alpha': [0.0, 0.7],
#     'reg_lambda': [0.0, 0.4]
}
# clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
# clf.fit(X_train, y_train)
# plot_grid_search(clf)
# table_grid_search(clf)


# # Grid search

# In[ ]:


lg_clf = LGBMClassifier(verbosity=VERBOSE,
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS)
lg_clf.min_child_weight = 1e-15
lg_clf.boosting_type = 'gbdt'
lg_clf.num_leaves = 55
lg_clf.max_depth = 21
lg_clf.learning_rate = 0.6
lg_clf.n_estimators = 2100
lg_clf.subsample_for_bin = 987
lg_clf.min_child_samples = 5
lg_clf.colsample_bytree = 0.9
lg_clf.reg_alpha = 0.0
lg_clf.reg_lambda = 0.0
# parameters = {
#     'max_depth': [15, 20, 25],
# }
parameters = {
    'boosting_type': ['gbdt', 'goss'],
    'n_estimators': [500, 1000, 1500, 2000],
    'num_leaves': [13, 21, 34, 55, 89, 144, 233, 377],
    'learning_rate': [0.5, 0.6, 0.7],
}
clf = GridSearchCV(lg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# ## Export grid search results

# In[ ]:


with open('clf.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# In[ ]:


clf.best_estimator_

