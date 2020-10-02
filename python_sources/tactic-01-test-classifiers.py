#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook checks all known classifiers with almost all their default parameters to see how good are they in this data set.
# 
# Some parameters have been configured to compare the models on equal terms:
# 
# The results are collected in [Tactic 99. Summary](https://www.kaggle.com/juanmah/tactic-99-summary).
# 
# The models in this notebook are, on purpose, not optimized.
# They will be optimized in successive notebooks in this tactic series,
# where some tactics will be tested and the results analysed.

# In[ ]:


import pip._internal as pip
pip.main(['install', '--upgrade', 'numpy==1.17.3'])
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels
from xgboost import XGBClassifier

import time

from lwoku import get_prediction


# # Define constants
# ## Set model parameters
# 
# - n_estimators = 2000. A big value to get a tight fit model.
# - min_sample_leafs = 100. To equalize all models.
# - random_state = 42. To get always the same results. And get always the same random split. 42 is the answer to the ultimate question of life, the universe, and everything.
# - n_jobs = -1. Use all processors.
# - verbose = 0. Per default, not nag this notebook. It could be change for testing.

# In[ ]:


N_ESTIMATORS = 2000
MIN_SAMPLE_LEAFS = 100
RANDOM_STATE = 42
N_JOBS = -1
VERBOSE = 0


# # Prepare data

# In[ ]:


# Read training and test files
X_train = pd.read_csv('../input/learn-together/train.csv', index_col='Id', engine='python')
X_test = pd.read_csv('../input/learn-together/test.csv', index_col='Id', engine='python')

# Define the dependent variable
y_train = X_train['Cover_Type'].copy()

# Define a training set
X_train = X_train.drop(['Cover_Type'], axis='columns')


# # Models
# ## Define

# ### [Generalized Linear Models](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
# #### [Logistic Regression classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# In[ ]:


lr_clf = LogisticRegression(verbose=VERBOSE,
                            random_state=RANDOM_STATE,
                            n_jobs=1)


# ### [Discriminant Analysis](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis)
# 
# #### [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis)

# In[ ]:


lda_clf = LinearDiscriminantAnalysis()


# ### [Nearest Neighbors](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
# 
# #### [k-nearest neighbors classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

# In[ ]:


knn_clf = KNeighborsClassifier(n_jobs=N_JOBS)


# ### [Naive Bayes](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)
# 
# #### [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)

# In[ ]:


gnb_clf = GaussianNB()


# ### [Support Vector Machines](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
# 
# #### [C-Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)

# In[ ]:


svc_clf = SVC(random_state=RANDOM_STATE,
              verbose=True)


# ### [Ensemble Methods](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
# 
# ### Bagging
# 

# #### [Bagging classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier)

# In[ ]:


bg_clf = BaggingClassifier(n_estimators=N_ESTIMATORS,
                           verbose=VERBOSE,
                           random_state=RANDOM_STATE)


# #### [Extra-trees classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier)

# In[ ]:


xt_clf = ExtraTreesClassifier(n_estimators=N_ESTIMATORS,
                              min_samples_leaf=MIN_SAMPLE_LEAFS,
                              verbose=VERBOSE,
                              random_state=RANDOM_STATE,
                              n_jobs=N_JOBS)


# #### [Random forest classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)

# In[ ]:


rf_clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                min_samples_leaf=MIN_SAMPLE_LEAFS,
                                verbose=VERBOSE,
                                random_state=RANDOM_STATE,
                                n_jobs=N_JOBS)


# ### Boosting

# #### [AdaBoost classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)

# In[ ]:


ab_clf = AdaBoostClassifier(n_estimators=N_ESTIMATORS,
                            random_state=RANDOM_STATE)


# #### [Gradient Boosting for classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)

# In[ ]:


gb_clf = GradientBoostingClassifier(n_estimators=N_ESTIMATORS,
                              min_samples_leaf=MIN_SAMPLE_LEAFS,
                              verbose=VERBOSE,
                              random_state=RANDOM_STATE)


# #### [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier)

# In[ ]:


lg_clf = LGBMClassifier(n_estimators=N_ESTIMATORS,
                        num_leaves=MIN_SAMPLE_LEAFS,
                        verbosity=VERBOSE,
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS)


# #### [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html?#xgboost.XGBClassifier)

# In[ ]:


xg_clf = XGBClassifier(random_state=RANDOM_STATE,
                       n_jobs=-N_JOBS,
                       learning_rate=0.1,
                       n_estimators=100,
                       max_depth=3)


# ### Model list

# In[ ]:


models = [
          ('lr', lr_clf),
          ('lda', lda_clf),
          ('knn', knn_clf),
          ('gnb', gnb_clf),
          ('svc', svc_clf),
          ('bg', bg_clf),
          ('xt', xt_clf),
          ('rf', rf_clf),
          ('ab', ab_clf),
          ('gb', gb_clf),
          ('lg', lg_clf),
          ('xg', xg_clf)
]


# In[ ]:


results = pd.DataFrame(columns = ['Model',
                                  'Accuracy',
                                  'Fit time',
                                  'Predict test set time',
                                  'Predict train set time'])

for name, model in models:

    # Fit
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    t_fit = (t1 - t0)
    
    # Predict test set
    t0 = time.time()
    y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
    t1 = time.time()
    t_test_pred = (t1 - t0)

    # Predict train set
    t0 = time.time()
    y_train_pred = pd.Series(get_prediction(model, X_train, y_train), index=X_train.index)
    accuracy = accuracy_score(y_train, y_train_pred)
    t1 = time.time()
    t_train_pred = (t1 - t0)

    # Submit
    y_train_pred.to_csv('train_' + name + '.csv', header=['Cover_Type'], index=True, index_label='Id')
    y_test_pred.to_csv('submission_' + name + '.csv', header=['Cover_Type'], index=True, index_label='Id')
    print('\n')
    
    results = results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Fit time': t_fit,
        'Predict test set time': t_test_pred,
        'Predict train set time': t_train_pred
    }, ignore_index = True)


# In[ ]:


results = results.sort_values('Accuracy', ascending=False).reset_index(drop=True)
results.to_csv('results.csv', index=True, index_label='Id')
results


# # Conclusions
# The best model with the default parameters,
# and without doing feature selection and feature engineering,
# is the Bagging classifier.
# 
# After checking all the models and choosing the best one (Bagging classifier).
# This one offers an improvement of about a 12 % in the accuracy respect the random forest classifier.
# 
# ## Adequacy of the models
# 
# Some models have taken into account, but finally, not added to this list. As not all the models are adequate for the data, or for the nature of the model.
# 
# ### Isolation Forest Algorithm
# 
# Returns the anomaly score of each sample, not the prediction.
# 
# It's useful for detecting outliers.
# 
# ### LightGBM
# 
# It's working extremely slowly (150 times slower) with numper 1.16.4 version.
# With 2000 estimators the notebook reach the 9 hours limit.
# 
# ### Ensemble of totally random trees
# 
# This model has no predict method.
# 
# ### Histogram-based Gradient Boosting Classification
# 
# This model is experimental, and it takes too long.

# # Do you know some other model that is not in this list?
# 
# Please, help me to find all the possible models (good or not) to analyse.
