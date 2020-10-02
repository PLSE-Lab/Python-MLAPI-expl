#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from gplearn.genetic import SymbolicRegressor,SymbolicTransformer
from gplearn.functions import make_function

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV

import os

print(os.listdir("../input"))
print(os.listdir("../input/LANL-Earthquake-Prediction"))
print(os.listdir("../input/lanl-features"))


# # Read Data

# In[ ]:


X = pd.read_csv('../input/lanl-features/train_features_denoised.csv')
X_test = pd.read_csv('../input/lanl-features/test_features_denoised.csv')
y = pd.read_csv('../input/lanl-features/y.csv')
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv',index_col='seg_id')


# # Scaling

# In[ ]:


X.drop('seg_id',axis=1,inplace=True)
X_test.drop('seg_id',axis=1,inplace=True)
X.drop('target',axis=1,inplace=True)
X_test.drop('target',axis=1,inplace=True)

alldata = pd.concat([X, X_test])

scaler = StandardScaler()

alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)

X = alldata[:X.shape[0]]
X_test = alldata[X.shape[0]:]


# # Feature Selection

# ## Drop highly correlated features

# In[ ]:


get_ipython().run_cell_magic('time', '', 'corr_matrix = X.corr()\ncorr_matrix = corr_matrix.abs()\nupper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))\nto_drop = [column for column in upper.columns if any(upper[column] > 0.95)]\n\nX = X.drop(to_drop, axis=1)\nX_test = X_test.drop(to_drop, axis=1)\nprint(X.shape)\nprint(X_test.shape)')


# ## Recursive feature elimination with cross validation and random forest regression

# Give training data some insight of the time range of this experiment. Little bit cheating

# In[ ]:


X["mean_y"] = np.full(len(y), y.values.mean())
X["max_y"] = np.full(len(y), y.values.max())
X["min_y"] = np.full(len(y), y.values.min())
X["std_y"] = np.full(len(y), y.values.std())

X_test["mean_y"] = np.full(len(X_test), y.values.mean())
X_test["max_y"] = np.full(len(X_test), y.values.max())
X_test["min_y"] = np.full(len(X_test), y.values.min())
X_test["std_y"] = np.full(len(X_test), y.values.std())

print(X.shape)
print(X_test.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', "rf = RandomForestRegressor(n_estimators = 10)\nrfecv = RFECV(estimator=rf, step=1, cv=3, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1) #3-fold cross-validation with mae\nrfecv = rfecv.fit(X, y.values)\nprint('Optimal number of features :', rfecv.n_features_)\nprint('Best features :', X.columns[rfecv.support_])\n\nX = X[X.columns[rfecv.support_].values]\nX_test = X_test[X_test.columns[rfecv.support_].values]\nprint(X.shape)\nprint(X_test.shape)")


# In[ ]:


print("Features:", list(X.columns))


# # Random Forest Hyperparameter Tuning

# To use RandomizedSearchCV, we first need to create a parameter grid to sample from during fitting:

# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


print("Grid Search Parameters", random_grid)


# <p>On each iteration, the algorithm will choose a difference combination of the features. Altogether, there are 2 \* 12 \* 2 \* 3 \* 3 \* 10 = 4320 settings! However, the benefit of a random search is that we are not trying every combination, but selecting at random to sample a wide range of values.</p>

# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X, y)


# The most important arguments in RandomizedSearchCV are n_iter, which controls the number of different combinations to try, and cv which is the number of folds to use for cross validation (we use 100 and 3 respectively). More iterations will cover a wider search space and more cv folds reduces the chances of overfitting, but raising each will increase the run time. Machine learning is a field of trade-offs, and performance vs time is one of the most fundamental.
# 
# We can view the best parameters from fitting the random search:

# In[ ]:


rf_random.best_params_


# ## Evaluate Random Search

# In[ ]:


def evaluate(model, features=X, labels=y):
    predictions = model.predict(features)    
    mae=mean_absolute_error(labels, predictions)
    print('Model Performance')
    print('Mean Absolute Error: {:0.4f}.'.format(mae))
    return mae


# ### Train a base model

# In[ ]:


base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X, y)
base_mae = evaluate(base_model, X, y)


# ### Train Best Random Model

# In[ ]:


best_random = rf_random.best_estimator_
random_mae = evaluate(best_random, X, y)


# In[ ]:


print('Improvement of {:0.2f}%.'.format(100 * (base_mae - random_mae) / base_mae))


# # Submission

# In[ ]:


submission.time_to_failure = best_random.predict(X_test)
submission.to_csv('submission.csv', index=True)


# In[ ]:


submission.head(10)

