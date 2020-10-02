#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to optimize the Extra-trees model.
# 
# First, all [Gradient Boosting for classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier) parameters are analysed separately.
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
from sklearn.ensemble import GradientBoostingClassifier

from lwoku import RANDOM_STATE, N_JOBS, VERBOSE, get_prediction
from grid_search_utils import plot_grid_search, table_grid_search

import pickle


# # Prepare data

# In[ ]:


# Read training and test files
X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')
X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')

# Define the dependent variable
y_train = X_train['Cover_Type'].copy()

# Define a training set
X_train = X_train.drop(['Cover_Type'], axis='columns')


# In[ ]:


VERBOSE=1


# # Search over parameters

# In[ ]:


gb_clf = GradientBoostingClassifier(verbose=VERBOSE,
                                    random_state=RANDOM_STATE)


# # loss
# ##### : {'deviance', 'exponential'}, optional (default='deviance')
# 
# loss function to be optimized. 'deviance' refers to
# deviance (= logistic regression) for classification
# with probabilistic outputs. For loss 'exponential' gradient
# boosting recovers the AdaBoost algorithm.
# 

# In[ ]:


parameters = {
    'loss': ['deviance']
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# Exponential value is not working. It gives the following message:
# 
# `ValueError: ExponentialLoss requires 2 classes; got 7 class(es)`

# # learning_rate
# ##### : float, optional (default=0.1)
# 
# learning rate shrinks the contribution of each tree by `learning_rate`.
# There is a trade-off between learning_rate and n_estimators.

# In[ ]:


parameters = {
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# The default value 0.1 and less don't give a good score.
# It increases for greater values,
# reaching the maximum at 0.3
# and then decreasing.

# # n_estimators
# ##### : int (default=100)
# 
# The number of boosting stages to perform. Gradient boosting
# is fairly robust to over-fitting so a large number usually
# results in better performance.

# In[ ]:


parameters = {
    'n_estimators': [100, 200, 500, 1000, 2000]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# The more estimators, the more the score is. Increasing the fit time proportionally as the number of estimators.

# # subsample
# #### : float, optional (default=1.0)
# 
# The fraction of samples to be used for fitting the individual base
# learners. If smaller than 1.0 this results in Stochastic Gradient
# Boosting. `subsample` interacts with the parameter `n_estimators`.
# Choosing `subsample < 1.0` leads to a reduction of variance
# and an increase in bias.
# 

# In[ ]:


parameters = {
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# It's a chaotic relation between subsample and score.

# # criterion
# ##### : string, optional (default="friedman_mse")
# 
# The function to measure the quality of a split. Supported criteria
# are "friedman_mse" for the mean squared error with improvement
# score by Friedman, "mse" for mean squared error, and "mae" for
# the mean absolute error. The default value of "friedman_mse" is
# generally the best as it can provide a better approximation in
# some cases.
# 

# In[ ]:


parameters = {
    'criterion': ['friedman_mse', 'mse']
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# `mse` criterion is a slightly better than `friedman_mse` criterior.
# `mae` criterion needs a huge amount of time (500 times more than the other two criteria).
# That is 3 hours for only one candidate.
# It has been removed because of time limit of the notebook.

# # min_samples_split
# ##### : int, float, optional (default=2)
# 
# The minimum number of samples required to split an internal node:
# - If int, then consider `min_samples_split` as the minimum number.
# - If float, then `min_samples_split` is a fraction and
#   `ceil(min_samples_split * n_samples)` are the minimum
#   number of samples for each split.

# In[ ]:


parameters = {
    'min_samples_split': [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The greater `min_samples_split` is, the lesser the score is.
# The greatest score value is for `min_samples_split` 5.

# # min_samples_leaf
# ##### : int, float, optional (default=1)
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
    'min_samples_leaf': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The greater `min_samples_leaf` is, the lesser the score is.
# The greatest score value is for `min_samples_leaf` 5.

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
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The default value has the greatest score.

# # max_depth
# ##### : integer, optional (default=3)
# 
# maximum depth of the individual regression estimators. The maximum
# depth limits the number of nodes in the tree. Tune this parameter
# for best performance; the best value depends on the interaction
# of the input variables.

# In[ ]:


parameters = {
    'max_depth': [1, 2, 5, 8, 13, 21, 34, 53, 54, 55, 89, None]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# From 21 `max_depth` estabilizes.
# It takes the greatest score with the value 13.
# Also is the one with the greatest fit time.

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
    'min_impurity_decrease': [x / 100 for x in range(0, 11)]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# It presents a chaotic behaviour.

# # init
# ##### : estimator or 'zero', optional (default=None)
# 
# An estimator object that is used to compute the initial predictions.
# ``init`` has to provide `fit` and `predict_proba`. If 'zero', the
# initial raw predictions are set to zero. By default, a
# ``DummyEstimator`` predicting the classes priors is used.

# In[ ]:


parameters = {
    'init': ['zero', None]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
table_grid_search(clf, all_ranks=True)


# None value has a greatest score.

# # max_features
# ##### : int, float, string or None, optional (default=None)
# 
# The number of features to consider when looking for the best split:
# 
# - If int, then consider `max_features` features at each split.
# - If float, then `max_features` is a fraction and
#   `int(max_features * n_features)` features are considered at each
#   split.
# - If "auto", then `max_features=sqrt(n_features)`.
# - If "sqrt", then `max_features=sqrt(n_features)`.
# - If "log2", then `max_features=log2(n_features)`.
# - If None, then `max_features=n_features`.
# 
# Choosing `max_features < n_features` leads to a reduction of variance
# and an increase in bias.
# 
# Note: the search for a split does not stop until at least one
# valid partition of the node samples is found, even if it requires to
# effectively inspect more than ``max_features`` features.

# In[ ]:


parameters = {
    'max_features': ['auto', 'sqrt', 'log2', 2, 5, 8, 13, 21, 34, None]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# The value of 34 has the greatest score, even more than None value.

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
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# The default value has the greatest score.
# Other values have a slightly less score,
# except 2 and 5 that are much less.

# # warm_start
# ##### : bool, default: False
# 
# When set to ``True``, reuse the solution of the previous call to fit
# and add more estimators to the ensemble, otherwise, just erase the
# previous solution.

# In[ ]:


parameters = {
    'warm_start': [True, False]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# This parameter doesn't have influence on the score.

# # presort
# ##### : bool or 'auto', optional (default='auto')
# 
# Whether to presort the data to speed up the finding of best splits in
# fitting. Auto mode by default will use presorting on dense data and
# default to normal sorting on sparse data. Setting presort to true on
# sparse data will raise an error.

# In[ ]:


parameters = {
    'presort': [True, False, 'auto']
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# As the data is dense, presorting gets a better score.
# Default value `auto` do presort.

# # validation_fraction
# ##### : float, optional, default 0.1
# 
# The proportion of training data to set aside as validation set for
# early stopping. Must be between 0 and 1.
# Only used if ``n_iter_no_change`` is set to an integer.

# In[ ]:


parameters = {
    'n_iter_no_change': [1],
    'validation_fraction': [x / 10 for x in range(1, 10)]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# # n_iter_no_change
# ##### : int, default None
# 
# ``n_iter_no_change`` is used to decide if early stopping will be used
# to terminate training when validation score is not improving. By
# default it is set to None to disable early stopping. If set to a
# number, it will set aside ``validation_fraction`` size of the training
# data as validation and terminate training when validation score is not
# improving in all of the previous ``n_iter_no_change`` numbers of
# iterations. The split is stratified.

# In[ ]:


parameters = {
    'n_iter_no_change': [1, 2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# This parameter doesn't have influence on the score.

# # tol
# ##### : float, optional, default 1e-4
# 
# Tolerance for the early stopping. When the loss is not improving
# by at least tol for ``n_iter_no_change`` iterations (if set to a
# number), the training stops.

# In[ ]:


parameters = {
    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# This parameter doesn't have influence on the score.

# # Exhaustive search

# In[ ]:


gb_clf.subsample = 0.8
gb_clf.min_samples_split = 5
gb_clf.min_samples_leaf = 5
gb_clf.max_depth = 13
gb_clf.min_impurity_decrease = 0.03
gb_clf.max_features = 34
parameters = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [500, 2000],
#     'subsample': [0.8, 0.9, 1.0],
    'criterion': ['friedman_mse', 'mse'],
#     'min_samples_split': [4, 5, 6],
#     'min_samples_leaf': [4, 5, 6],
#     'max_depth': [12, 13, 14],
    'min_impurity_decrease': [0, 0.03],
#     'max_features': [21, 34, None]
}
clf = GridSearchCV(gb_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# ## Export grid search results

# In[ ]:


with open('clf.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# In[ ]:


clf.best_estimator_

