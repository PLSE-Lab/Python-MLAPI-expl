#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to optimize the Logistic Regression model.
# 
# First, all [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis) parameters are analysed separately.
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

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


lda_clf = LinearDiscriminantAnalysis()


# # solver
# ##### : string, optional
# 
# Solver to use, possible values:
#   - 'svd': Singular value decomposition (default).
#     Does not compute the covariance matrix, therefore this solver is
#     recommended for data with a large number of features.
#   - 'lsqr': Least squares solution, can be combined with shrinkage.
#   - 'eigen': Eigenvalue decomposition, can be combined with shrinkage.

# In[ ]:


parameters = {
    'solver': ['svd', 'lsqr'] # eigen solver fails
}
clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The solver `eigen` fails. The both working solvers has the same score. 

# # shrinkage
# ##### : string or float, optional
# 
# > Shrinkage parameter, possible values:
# >   - None: no shrinkage (default).
# >   - 'auto': automatic shrinkage using the Ledoit-Wolf lemma.
# >   - float between 0 and 1: fixed shrinkage parameter.
# > Note that shrinkage works only with 'lsqr' and 'eigen' solvers.

# In[ ]:


parameters = {
    'solver': ['lsqr'],
    'shrinkage': [None] + [x / 10 for x in range(0, 11)] + ['auto']
}
clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The solver `lsqr` with  shrinkage `auto` has a slightly better score.

# # priors
# ##### : array, optional, shape (n_classes,)
# 
# Class priors.
# 
# **Not applied**

# # n_components
# ###### : int, optional (default=None)
# 
# Number of components (<= min(n_classes - 1, n_features)) for
# dimensionality reduction. If None, will be set to
# min(n_classes - 1, n_features).

# In[ ]:


parameters = {
    'n_components': [None] + [1, 2, 5, 8, 13, 21, 34, 55]
}
clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The score is equal regardless the number of components. The fit time is the same too.

# # store_covariance
# ##### : bool, optional
# 
# Additionally compute class covariance matrix (default False), used
# ![](http://)only in 'svd' solver.

# In[ ]:


parameters = {
    'solver': ['svd'],
    'store_covariance': [True, False]
}
clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The parameter `store_covariance` has no effect in the score.

# # tol
# ##### : float, optional, (default 1.0e-4)
# 
# Threshold used for rank estimation in SVD solver.

# In[ ]:


parameters = {
    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
}
clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The parameter `tol` has no effect in the score, neither the fit time.

# # Exhaustive search

# In[ ]:


parameters = {
    'solver': ['svd', 'lsqr'],
    'n_components': [None] + [1, 2, 5, 8, 13, 21, 34, 55],
    'store_covariance': [True, False],
    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
}
clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# The better solver without shrinkage is `lsqr`.
# 
# The parameters: `n_components`, `store_covariance` and `tol`, have no effect in the score.

# To try the effect of shrinkage, a grid search with fixed solver `lsqr` is performed.

# In[ ]:


parameters = {
    'solver': ['lsqr'],
    'shrinkage': [None] + [x / 10 for x in range(0, 11)] + ['auto'],
    'n_components': [None] + [1, 2, 5, 8, 13, 21, 34, 55],
    'store_covariance': [True, False],
    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
}
clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# Shrinkage `auto` gets a better score.
# 
# The parameters: `n_components`, `store_covariance` and `tol`, have no effect in the score.

# In[ ]:


parameters = {
    'solver': ['lsqr'],
    'shrinkage': [None] + [x / 10 for x in range(0, 11)] + ['auto']
}
clf = GridSearchCV(lda_clf, parameters, cv=5, verbose=VERBOSE, n_jobs=N_JOBS)
clf.fit(X_train, y_train)
plot_grid_search(clf)
table_grid_search(clf)


# ## Export grid search results

# In[ ]:


with open('clf.pickle', 'wb') as fp:
    pickle.dump(clf, fp)


# # Conclusion

# The best results are get with solver `lsqr` and shrinkage `auto`.
# All other parameters has no effect in getting a better score.

# In[ ]:


lda_clf = clf.best_estimator_
lda_clf

