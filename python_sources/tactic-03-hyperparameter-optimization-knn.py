#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to optimize the Logistic Regression model.
# 
# First, all [k-nearest neighbors classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) parameters are analysed separately.
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
from sklearn.neighbors import KNeighborsClassifier

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


knn_clf = KNeighborsClassifier(n_jobs=N_JOBS)


# # n_neighbors
# ##### : int, optional (default = 5)
# 
# Number of neighbors to use by default for :meth:`kneighbors` queries.

# In[ ]:


parameters = {
    'n_neighbors': range(1, 11, 1)
}
clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The best score is with 1 neighbor, and the score decreases with the number of neighbors.

# # weights
# ##### : str or callable, optional (default = 'uniform')
# 
# weight function used in prediction.  Possible values:
# - 'uniform' : uniform weights.  All points in each neighborhood
# are weighted equally.
# - 'distance' : weight points by the inverse of their distance.
# in this case, closer neighbors of a query point will have a
# greater influence than neighbors which are further away.
# - [callable] : a user-defined function which accepts an
# array of distances, and returns an array of the same shape
# containing the weights.

# In[ ]:


parameters = {
    'weights': ['uniform', 'distance']
}
clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Distance weight score more than uniform one.

# # algorithm
# ##### : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
# 
# Algorithm used to compute the nearest neighbors:
# - 'ball_tree' will use :class:`BallTree`
# - 'kd_tree' will use :class:`KDTree`
# - 'brute' will use a brute-force search.
# - 'auto' will attempt to decide the most appropriate algorithm
#   based on the values passed to :meth:`fit` method.
# Note: fitting on sparse input will override the setting of
# this parameter, using brute force.

# In[ ]:


parameters = {
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# All algorithms score the same.
# The brute is the one that needs the least fit time, but biggest score time.

# # leaf_size
# ##### : int, optional (default = 30)
# 
# Leaf size passed to BallTree or KDTree. This can affect the
# speed of the construction and query, as well as the memory
# required to store the tree.  The optimal value depends on the
# nature of the problem.

# In[ ]:


parameters = {
    'leaf_size': range(20, 50, 5)
}
clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The leaf size doesn't affect the score.

# # p
# ##### : integer, optional (default = 2)
# 
# Power parameter for the Minkowski metric. When p = 1, this is
# equivalent to using manhattan_distance (l1), and euclidean_distance
# (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

# In[ ]:


parameters = {
    'p': range(1, 4)
}
clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Manhattan distance is the value that scores more.

# # metric
# ##### : string or callable, default 'minkowski'
# 
# the distance metric to use for the tree.  The default metric is
# minkowski, and with p=2 is equivalent to the standard Euclidean
# metric. See the documentation of the DistanceMetric class for a
# list of available metrics.
# 
# **Note**: Not evaluated

# # metric_params
# ##### : dict, optional (default = None)
# 
# Additional keyword arguments for the metric function.
# 
# **Note**: Not evaluated

# # Exhaustive search

# In[ ]:


parameters = {
    'n_neighbors': range(1, 11, 1),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': range(20, 50, 5),
    'p': range(1, 4)
}
clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# it seems that `leaf_size` has no effect on the score.
# 
# A new grid search is performed without it.

# In[ ]:


parameters = {
    'n_neighbors': range(1, 11, 1),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': range(1, 4)
}
clf = GridSearchCV(knn_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# ## Export grid search results

# In[ ]:


with open('clf.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# In[ ]:


clf.best_estimator_

