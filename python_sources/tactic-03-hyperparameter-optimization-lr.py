#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to optimize the Logistic Regression model.
# 
# First, all [LogisticRegression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) parameters are analysed separately.
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

from sklearn.linear_model import LogisticRegression
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


lr_clf = LogisticRegression(verbose=VERBOSE,
                            random_state=RANDOM_STATE,
                            n_jobs=N_JOBS)


# # penalty
# ##### : str, 'l1', 'l2', 'elasticnet' or 'none', optional (default='l2')
# 
# Used to specify the norm used in the penalization*. The 'newt*on-cg',
# 'sag' and 'lbfgs' solvers support only l2 penalties. 'elasticnet' is
# only supported by the 'saga' solver. If 'none' (not supported by the
# liblinear solver), no regularization is applied.

# In[ ]:


parameters = {
    'solver': ['newton-cg', 'sag', 'lbfgs'],
    'penalty': ['none', 'l2']
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# The solver `newton-cg` scores more with no penalty.
# Whereas `lbfgs` scores more with `l2` penalty.
# The solver `sag` has the same score.

# In[ ]:


parameters = {
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2']
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# The solver `liblinear` scores more with `l1` penalty.

# In[ ]:


parameters = {
    'solver': ['saga'],
    'l1_ratio': [x / 10 for x in range(0, 11)],
    'penalty': ['none', 'elasticnet']
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The penalty `elasticnet` has no effect on score for whatever value of `l1_ratio`.
# Fit time is better for no penalty.

# # dual
# ##### : bool, optional (default=False)
# 
# Dual or primal formulation. Dual formulation is only implemented for
# l2 penalty with liblinear solver. Prefer dual=False when
# n_samples > n_features.

# In[ ]:


parameters = {
    'solver': ['liblinear'],
    'penalty': ['l2'],
    'dual': [True, False]
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf, all_ranks=True)


# Default value False is the best option.

# # tol
# ##### : float, optional (default=1e-4)
# 
# Tolerance for stopping criteria.

# In[ ]:


parameters = {
    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# From `1e-08` the score already has the maximum value. The minimum fit time for the maximum score if also for `1e-08`.

# #    C
# ###### : float, optional (default=1.0)
# 
# Inverse of regularization strength; must be a positive float.
# Like in support vector machines, smaller values specify stronger
# regularization.

# In[ ]:


parameters = {
    'C': [(0.9 + x / 50) for x in range(0, 10)]
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# There are two values with the same score: 1 and 1.08.
# It presents a chaotic behaviour.

# # fit_intercept
# ##### : bool, optional (default=True)
# 
# Specifies if a constant (a.k.a. bias or intercept) should be
# added to the decision function.

# In[ ]:


parameters = {
    'fit_intercept': [True, False]
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The parameter `fit_intercept` with `True` value has a slightly better score.

# # intercept_scaling
# ##### : float, optional (default=1)
# 
# Useful only when the solver 'liblinear' is used
# and self.fit_intercept is set to True. In this case, x becomes
# [x, self.intercept_scaling],
# i.e. a "synthetic" feature with constant value equal to
# intercept_scaling is appended to the instance vector.
# The intercept becomes ``intercept_scaling * synthetic_feature_weight``.
# 
# Note! the synthetic feature weight is subject to l1/l2 regularization
# as all other features.
# To lessen the effect of regularization on synthetic feature weight
# (and therefore on the intercept) intercept_scaling has to be increased.

# In[ ]:


parameters = {
    'solver' : ['liblinear'],
    'fit_intercept': [True],
    'intercept_scaling': [1, 2, 3, 5, 8, 13, 21, 34]
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Intercept_scaling presents a chaotic behaviour.
# The value 13 has the greatest score.

# # class_weight
# ##### : dict or 'balanced', optional (default=None)
# 
# Weights associated with classes in the form ``{class_label: weight}``.
# If not given, all classes are supposed to have weight one.
# 
# The "balanced" mode uses the values of y to automatically adjust
# weights inversely proportional to class frequencies in the input data
# as ``n_samples / (n_classes * np.bincount(y))``.
# 
# Note that these weights will be multiplied with sample_weight (passed
# through the fit method) if sample_weight is specified.

# All classes in train are equal, then 'balanced' value has no sense.
# 
# The weight of the classes in the test set can be estimated from the results of a good prediction.
# The LightGBM model has a 0.77311 of public score.
# This is a good approximation.
# The real weighting will differ.

# In[ ]:


parameters = {
    'class_weight' : [None, 'balanced']
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
table_grid_search(clf, all_ranks=True)


# As all the categories has the same number of samples,
# `balanced` and `None` options has the same result.

# #     solver
# ##### : str, {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, optional (default='liblinear').
# 
# Algorithm to use in the optimization problem.
# 
# - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
#   'saga' are faster for large ones.
# - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
#   handle multinomial loss; 'liblinear' is limited to one-versus-rest
#   schemes.
# - 'newton-cg', 'lbfgs', 'sag' and 'saga' handle L2 or no penalty
# - 'liblinear' and 'saga' also handle L1 penalty
# - 'saga' also supports 'elasticnet' penalty
# - 'liblinear' does not handle no penalty
# 
# Note that 'sag' and 'saga' fast convergence is only guaranteed on
# features with approximately the same scale. You can
# preprocess the data with a scaler from sklearn.preprocessing.

# In[ ]:


parameters = {
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# `newton-cg` has clearly the greatest score, and also the greatest fit time.

# #     max_iter
# ##### : int, optional (default=100)
# 
#         Maximum number of iterations taken for the solvers to converge.

# In[ ]:


parameters = {
    'max_iter': range(50, 250, 50)
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# # multi_class
# ##### : str, {'ovr', 'multinomial', 'auto'}, optional (default='ovr')
# 
# If the option chosen is 'ovr', then a binary problem is fit for each
# label. For 'multinomial' the loss minimised is the multinomial loss fit
# across the entire probability distribution, *even when the data is
# binary*. 'multinomial' is unavailable when solver='liblinear'.
# 'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
# and otherwise selects 'multinomial'.

# In[ ]:


parameters = {
    'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
    'multi_class': ['ovr', 'multinomial']
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Multinomial is better for `newton-cg`, `sag` and `saga`.
# Where as ovr is better for `lbfgs`.

# # warm_start
# ##### : bool, optional (default=False)
# 
# When set to True, reuse the solution of the previous call to fit as
# initialization, otherwise, just erase the previous solution.
# Useless for liblinear solver.

# In[ ]:


parameters = {
    'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
    'warm_start': [True, False]
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Warm_start seems to have no effect.

# # l1_ratio
# ##### : float or None, optional (default=None)
# 
# The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
# used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
# to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
# to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
# combination of L1 and L2.

# Used in combination with penalty='elasticnet'.

# # Exhaustive search

# Eash solver has some compatible values at some parameters.
# Thus, a complete grid search could not be done.
# 
# The strategy is to do a grid search for each cluster of compatible solvers.

# ## liblinear

# In[ ]:


parameters = {
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2'],
    'C': [0.98, 1.00, 1.02],
    'tol': [1e-7, 1e-8, 1e-9],
    'max_iter': range(100, 250, 50)
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# ### Export grid search results

# In[ ]:


with open('clf_liblinear.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# ## saga

# In[ ]:


parameters = {
    'solver': ['saga'],
    'max_iter': range(100, 250, 50)
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# ### Export grid search results

# In[ ]:


with open('clf_saga.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# ## sag

# In[ ]:


parameters = {
    'solver': ['sag'],
    'max_iter': range(100, 250, 50),
    'multi_class': ['ovr', 'multinomial'],
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# ### Export grid search results

# In[ ]:


with open('clf_sag.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# ## lbfgs

# In[ ]:


parameters = {
    'solver': ['lbfgs'],
    'penalty': ['none', 'l2'],
    'C': [0.98, 1.00, 1.02],
    'fit_intercept': [True, False],
    'max_iter': range(100, 250, 50),
    'multi_class': ['ovr', 'multinomial'],
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# ### Export grid search results

# In[ ]:


with open('clf_lbfgs.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# ## newton-cg

# In[ ]:


parameters = {
    'solver': ['newton-cg'],
    'penalty': ['none', 'l2'],
    'C': [0.98, 1.00, 1.02],
    'fit_intercept': [True, False],
    'max_iter': range(100, 250, 50),
    'multi_class': ['ovr', 'multinomial'],
}
clf = GridSearchCV(lr_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# ### Export grid search results

# In[ ]:


with open('clf_newton-cg.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# # Conclusions

# Each solver is itself an own model with its own particularities.
# Each parameter affect in a different way each solver.
# Thus, all solvers has analized separately.
# 
# Improving each solver separately make them better than at the beginning,
# but each solver has the same rank in the maximum score.
# 
# The best scoring solvers are newton-cg and liblinear.
