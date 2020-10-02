#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to optimize the Random Forest model.
# 
# First, all [Random forest classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) parameters are analysed separately.
# 
# Then, a grid search is carried out.
# This is a search through all the combinations of parameters,
# which optimize the internal score in the train set.
# 
# The results are collected at [Tactic 03. Hyperparameter optimization](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization).

# In[ ]:


import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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


rf_clf = RandomForestClassifier(verbose=VERBOSE,
                                random_state=RANDOM_STATE,
                                n_jobs=N_JOBS)


# # n_estimators
# ##### : integer, optional (default=10)
# 
# The number of trees in the forest.

# In[ ]:


parameters = {
    'n_estimators': [200, 300, 400, 500, 550, 600, 650, 700, 750, 800, 900]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The optimal value is near 700 estimators. Until this value, there is a progressive increase, with the exception of an abnormally low value at 600 estimators. After 700 estimators, it plummets. 

# # criterion
# ##### : string, optional (default="gini")
# 
# The function to measure the quality of a split. Supported criteria are
# "gini" for the Gini impurity and "entropy" for the information gain.
# Note: this parameter is tree-specific.

# In[ ]:


parameters = {
    'criterion': ['gini', 'entropy']
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The `gini` value offers a slightly better result.

#  # max_depth
#  ##### : integer or None, optional (default=None)
#  
# The maximum depth of the tree. If None, then nodes are expanded until
# all leaves are pure or until all leaves contain less than
# min_samples_split samples.

# In[ ]:


parameters = {
    'max_depth': [10, 20, 25, 30, 35, 40, 50, 60, None]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# Values higher than 20 have a good score.
# The value of 30 has an outstanding good score (more than None).

# # min_samples_split
# ##### : int, float, optional (default=2)
# 
# The minimum number of samples required to split an internal node:
# 
# - If int, then consider `min_samples_split` as the minimum number.
# - If float, then `min_samples_split` is a fraction and
#   `ceil(min_samples_split * n_samples)` are the minimum
#   number of samples for each split.
# 

# In[ ]:


parameters = {
    'min_samples_split': [2, 3, 4, 5, 8, 13, 21, 34, 55, 89]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Forcing not to split when there is a low number of samples, makes the score descent.
# And maybe could avoid overfit.

# # min_samples_leaf
# ###### : int, float, optional (default=1)
# 
# The minimum number of samples required to be at a leaf node.
# A split point at any depth will only be considered if it leaves at
# least ``min_samples_leaf`` training samples in each of the left and
# right branches.  This may have the effect of smoothing the model,
# especially in regression.
# 
# - If int, then consider `min_samples_leaf` as the minimum number.
# - If float, then `min_samples_leaf` is a fraction and
#   `ceil(min_samples_leaf * n_samples)` are the minimum
#   number of samples for each node.

# In[ ]:


parameters = {
    'min_samples_leaf': [1, 2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Preventing to have leafs with low number of samples, makes the score descent.
# And maybe could avoid overfit.

# # min_weight_fraction_leaf
# ##### : float, optional (default=0.)
# 
# The minimum weighted fraction of the sum total of weights (of all
# the input samples) required to be at a leaf node. Samples have
# equal weight when sample_weight is not provided.

# In[ ]:


parameters = {
    'min_weight_fraction_leaf': [x / 10 for x in range(0, 6)]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The best score is for the default value.

# # max_features
# ##### : int, float, string or None, optional (default="auto")
# 
# The number of features to consider when looking for the best split:
# 
# - If int, then consider `max_features` features at each split.
# - If float, then `max_features` is a fraction and
#   `int(max_features * n_features)` features are considered at each
#   split.
# - If "auto", then `max_features=sqrt(n_features)`.
# - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
# - If "log2", then `max_features=log2(n_features)`.
# - If None, then `max_features=n_features`.
# 
# Note: the search for a split does not stop until at least one
# valid partition of the node samples is found, even if it requires to
# effectively inspect more than ``max_features`` features.

# In[ ]:


parameters = {
    'max_features': ['auto', 'sqrt', 'log2', 2, 5, 8, 10, 13, 21, 34, None]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# The better score is for `max_features` 13.
# It's a local maximum, before and after, the scores decreases.

# # max_leaf_nodes
# ##### : int or None, optional (default=None)
# 
# Grow trees with ``max_leaf_nodes`` in best-first fashion.
# Best nodes are defined as relative reduction in impurity.
# If None then unlimited number of leaf nodes.

# In[ ]:


parameters = {
    'max_leaf_nodes': [2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, None]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# More leaf nodes, more score. None has unlimited leaf nodes and has the maximum score.

# # min_impurity_decrease
# ##### : float, optional (default=0.)
# 
# A node will be split if this split induces a decrease of the impurity
# greater than or equal to this value.
# 
# The weighted impurity decrease equation is the following::
# 
#     N_t / N * (impurity - N_t_R / N_t * right_impurity
#                         - N_t_L / N_t * left_impurity)
#                         
# where ``N`` is the total number of samples, ``N_t`` is the number of
# samples at the current node, ``N_t_L`` is the number of samples in the
# left child, and ``N_t_R`` is the number of samples in the right child.
# 
# ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
# if ``sample_weight`` is passed.

# In[ ]:


parameters = {
    'min_impurity_decrease': [x / 10 for x in range(0, 11)]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The best score is with the default value.

# # bootstrap
# ##### : boolean, optional (default=True)
# 
# Whether bootstrap samples are used when building trees. If False, the
# whole datset is used to build each tree.
# 

# In[ ]:


parameters = {
    'bootstrap': [True, False]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The best score is with the default value.

# # oob_score
# ##### : bool (default=False)
#   
# Whether to use out-of-bag samples to estimate
# the generalization accuracy.

# In[ ]:


parameters = {
    'oob_score': [True, False]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The score is the same. The fit time is greater when it's true. The score is less than the score with bootstrap equal to true.

# # warm_start
# ##### : bool, optional (default=False)
# 
# When set to ``True``, reuse the solution of the previous call to fit
# and add more estimators to the ensemble, otherwise, just fit a whole
# new forest.

# In[ ]:


parameters = {
    'warm_start': [True, False]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# It has the same score.

# # class_weight
# ##### : dict, list of dicts, "balanced", "balanced_subsample" or None, optional (default=None)
# 
# Weights associated with classes in the form ``{class_label: weight}``.
# If not given, all classes are supposed to have weight one. For
# multi-output problems, a list of dicts can be provided in the same
# order as the columns of y.
# 
# Note that for multioutput (including multilabel) weights should be
# defined for each class of every column in its own dict. For example,
# for four-class multilabel classification weights should be
# [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
# [{1:1}, {2:5}, {3:1}, {4:1}].
# 
# The "balanced" mode uses the values of y to automatically adjust
# weights inversely proportional to class frequencies in the input data
# as ``n_samples / (n_classes * np.bincount(y))``
# 
# The "balanced_subsample" mode is the same as "balanced" except that
# weights are computed based on the bootstrap sample for every tree
# grown.
# 
# For multi-output, the weights of each column of y will be multiplied.
# 
# Note that these weights will be multiplied with sample_weight (passed
# through the fit method) if sample_weight is specified.

# In[ ]:


parameters = {
    'class_weight': ['balanced', 'balanced_subsample', None]
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
# plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# As all the categories has the same number of samples,
# `balanced` and `None` options has the same result.

# # Exhaustive search

# In[ ]:


parameters = {
    'n_estimators': [650, 700],
    'max_depth': [30, None],
    'max_features': [13, 21, 34, None],
    'bootstrap': [True, False],
}
clf = GridSearchCV(rf_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# These four parameters are better at some value different as its default value.
# The number of estimators turns out to be better 700 than 10, `max_depth` 30 than None, `max_features` 21 than None, and `bootstrap` False than True.

# ## Export grid search results

# In[ ]:


with open('clf.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# In[ ]:


clf.best_estimator_

