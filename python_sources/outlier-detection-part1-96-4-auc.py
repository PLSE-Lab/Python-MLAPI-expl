#!/usr/bin/env python
# coding: utf-8

# In this notebook I'm exploring unsupervised methods for fraud detection. 
# 
# The first part shows hyperparameter optimization for Isolation Forest with the best set of parameters {'max_features': 18, 'n_estimators': 30} and area under the ROC curve: 96.4353023629.
# 
# Coming soon: 
# Experiments with Elliptic Envelope :)

# In[ ]:


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets.mldata import fetch_mldata
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer, roc_auc_score
import scipy as sp


# In[ ]:


data = pd.read_csv("../input/creditcard.csv")

X = data.drop(["Time", "Amount"], axis=1)
X = X.as_matrix()
y = X[:, -1]


# In[ ]:


IsolationForest().get_params()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X[:, :-1], y, test_size=0.2, random_state=777)

pipeline = IsolationForest()

param_grid = {'n_estimators': range(10, 50, 10), "max_features": range(8, 29, 10)}

ss = ShuffleSplit(test_size=0.2, random_state=777)

def scorer(pipeline, X, y):
    y_score = - pipeline.decision_function(X)
    score = roc_auc_score(y, y_score) * 100.
    print (score)
    return score

grid = GridSearchCV(pipeline, param_grid, cv=ss, scoring = scorer)

grid.fit(X, y)

print("Grid scores on development set:")
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

print("Best parameters set found on development set:")
print(grid.best_params_)
print("ROC AUC of the best estimator: ")
print(grid.best_score_)

