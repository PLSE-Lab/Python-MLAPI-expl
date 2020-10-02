#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to optimize the Logistic Regression model.
# 
# First, all [Bagging classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier) parameters are analysed separately.
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
from sklearn.ensemble import BaggingClassifier

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


# # Search over parameters

# In[ ]:


bg_clf = BaggingClassifier(verbose=VERBOSE,
                           random_state=RANDOM_STATE,
                           n_jobs=N_JOBS)


# # base_estimator
# ##### : object or None, optional (default=None)
# 
# The base estimator to fit on random subsets of the dataset.
# If None, then the base estimator is a decision tree.

# **Note**: Not evaluated

# # n_estimators
# ##### : int, optional (default=10)
# 
# The number of base estimators in the ensemble.

# In[ ]:


parameters = {
    'n_estimators': [20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
}
clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The score increases as the n_estimator does too.
# From 200 estimators, the score estabilizes, with some noise.
# The fit and score times incresase as well.

# # max_samples
# ##### : int or float, optional (default=1.0)
# 
# The number of samples to draw from X to train each base estimator.
# - If int, then draw `max_samples` samples.
# - If float, then draw `max_samples * X.shape[0]` samples.

# In[ ]:


parameters = {
    'max_samples': [x / 10 for x in range(1, 11)]
}
clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# While with half of samples it is already a good approximation,
# the maximum score is achieved then all samples are taken into account.

# # max_features
# ##### : int or float, optional (default=1.0)
# 
# The number of features to draw from X to train each base estimator.
# - If int, then draw `max_features` features.
# - If float, then draw `max_features * X.shape[1]` features.

# In[ ]:


parameters = {
    'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.92, 0.95, 1.0]
}
clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# More features, more score.
# But it decays near the end.
# The optimal point is with the 92 % of the features.

# # bootstrap
# ##### : boolean, optional (default=True)
# 
# Whether samples are drawn with replacement. If False, sampling
# without replacement is performed.

# In[ ]:


parameters = {
    'bootstrap': [True, False]
}
clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# When samples are drawn with replacement, the score is higher.

# # bootstrap_features
# ##### : boolean, optional (default=False)
# 
# Whether features are drawn with replacement.

# In[ ]:


parameters = {
    'bootstrap_features': [True, False]
}
clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# When features are drawn with replacement, the score is higher.

# # oob_score
# ##### : bool, optional (default=False)
# 
# Whether to use out-of-bag samples to estimate
# the generalization error.

# In[ ]:


parameters = {
    'oob_score': [True, False]
}
clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# There is no difference in score when using out-of-bag samples.
# Instead, the fit and score times are higher.

# # warm_start
# ##### : bool, optional (default=False)
# 
# When set to True, reuse the solution of the previous call to fit
# and add more estimators to the ensemble, otherwise, just fit
# a whole new ensemble.

# In[ ]:


parameters = {
    'warm_start': [True, False]
}
clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# There is no difference in score when using `warm_start`.
# Instead, the fit and score times are higher.

# # Exhaustive search

# In[ ]:


parameters = {
    'n_estimators': [300, 400, 500, 600, 700, 800],
    'max_features': [0.90, 0.92, 0.95, 1.0],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
}
clf = GridSearchCV(bg_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The best results are achieved with `bootstrap` `False` and `bootstrap_features` `True`.
# 
# The score is better with `max_features` 0.92 and `n_estimators` 500.

# ## Export grid search results

# In[ ]:


with open('clf.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# In[ ]:


clf.best_estimator_

