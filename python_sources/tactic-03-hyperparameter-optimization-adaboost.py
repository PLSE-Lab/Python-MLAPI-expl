#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to optimize the Extra-trees model.
# 
# First, all [AdaBoost classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier) parameters are analysed separately.
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
from sklearn.ensemble import AdaBoostClassifier

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


ab_clf = AdaBoostClassifier(random_state=RANDOM_STATE)


# # base_estimator
# ##### : object, optional (default=None)
# 
# The base estimator from which the boosted ensemble is built.
# Support for sample weighting is required, as well as proper
# ``classes_`` and ``n_classes_`` attributes. If ``None``, then
# the base estimator is ``DecisionTreeClassifier(max_depth=1)``

# **Note**: Not evaluated

# # n_estimators
# ##### : integer, optional (default=50)
# 
# The maximum number of estimators at which boosting is terminated.
# In case of perfect fit, the learning procedure is stopped early.
# 

# In[ ]:


parameters = {
    'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30]
}
clf = GridSearchCV(ab_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The relation between the estimators and the score is pretty chaotic.
# The best score is with 6 estimators.
# An even number scores more than an odd number.

# # learning_rate
# ##### : float, optional (default=1.)
# 
# Learning rate shrinks the contribution of each classifier by
# ``learning_rate``. There is a trade-off between ``learning_rate`` and
# ``n_estimators``.
# 

# In[ ]:


parameters = {
    'learning_rate': [(0.97 + x / 100) for x in range(0, 8)]
}
clf = GridSearchCV(ab_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# There's no visible relation between `learning_rate` and score.

# # algorithm
# ##### : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
# 
# If 'SAMME.R' then use the SAMME.R real boosting algorithm.
# ``base_estimator`` must support calculation of class probabilities.
# If 'SAMME' then use the SAMME discrete boosting algorithm.
# The SAMME.R algorithm typically converges faster than SAMME,
# achieving a lower test error with fewer boosting iterations.

# In[ ]:


parameters = {
    'algorithm': ['SAMME', 'SAMME.R']
}
clf = GridSearchCV(ab_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# SAMME algorithm seems to have better score.

# # Exhaustive search

# In[ ]:


parameters = {
    'n_estimators': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
    'learning_rate': [(0.97 + x / 100) for x in range(0, 8)],
    'algorithm': ['SAMME', 'SAMME.R']
}
clf = GridSearchCV(ab_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The previous assumptions have not been very successful, since the parameters depend on each other.

# ## Export grid search results

# In[ ]:


with open('clf.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# In[ ]:


clf.best_estimator_

