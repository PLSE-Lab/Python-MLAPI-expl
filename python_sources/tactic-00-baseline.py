#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook establishes a baseline to test tactics and see if they improve the score.
# The results are collected in [Tactic 99. Summary](https://www.kaggle.com/juanmah/tactic-99-summary).
# 
# The model choosen in this notebook is the Random Forest Classifier.
# 
# This model is, on purpose, not optimized.
# It will be optimized in successive notebooks in this tactic series,
# where some tactics will be tested and the results analysed.

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from lwoku import get_accuracy, get_prediction, plot_confusion_matrix, plot_features_importance


# # Define constants

# ## Set model parameters

# In[ ]:


RANDOM_STATE = 1
N_JOBS = -1
N_ESTIMATORS = 2000


# # Prepare data

# ## Read training and test files

# In[ ]:


X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')
X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')


# ## Define the dependent variable

# In[ ]:


y_train = X_train['Cover_Type'].copy()


# ## Define a training set

# In[ ]:


X_train = X_train.drop(['Cover_Type'], axis='columns')


# # Model

# ## Define

# In[ ]:


rf_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                min_samples_leaf=100,
                                verbose=1,
                                random_state=RANDOM_STATE,
                                n_jobs=N_JOBS)


# ## Fit

# In[ ]:


rf_clf.fit(X_train, y_train)


# ## Predict

# In[ ]:


y_test_pred = pd.Series(rf_clf.predict(X_test), index=X_test.index)


# ## Predict for train set

# In[ ]:


y_train_pred = get_prediction(rf_clf, X_train, y_train)


# ## Accuracy

# In[ ]:


accuracy_score(y_train, y_train_pred)


# ## Plot confusion matrix

# In[ ]:


plot_confusion_matrix(y_train, y_train_pred)


# ## Plot features importance

# In[ ]:


plot_features_importance(X_train.columns, rf_clf)


# # Create submission

# In[ ]:


y_test_pred.to_csv('submission.csv', header=['Cover_Type'], index=True, index_label='Id')

