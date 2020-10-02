#!/usr/bin/env python
# coding: utf-8

# **Algorithms Comparison**
# 
# We compare 4 algorithms:
# 
# * KNN
# * Random forests
# * Naive Bayes
# * Adaboost ensemble
# 
# The training set is used for nested cross validation, in order to estimate the performance of each algorithm on unseen data, based on the average cross validation scores of accuracy.
# 
# The nested cross validation procedure consists of two loops [1]:
# 
# *  an inner k-fold loop (k=2, stratified folding) that splits the data into training and validation sets and selects the best model parameters
# *  an outer k-fold loop (k = 5) that splits the data into training and test folds
# 
# The outer loop employs Scikit's method *cross_validate*, providing the metric scores per fold. We calculate the average and standard deviation of the results for each algorithm.<br>
# 
# The inner loop employs Scikit's *RandomizedSearchCV* to select the best model parameters drawn from specified distributions. In this exercise, *RandomizedSearchCV* is preferred over *GridSearchCV*, in order to speed up the tuning process.<br>
# 
# Stratified folding is used on the outer loop to preserve the ratio of the classes in each fold.<br>
# 
# Regarding data processing, we use Scikit's *Pipelines()* that include both preprocessing steps, such as scaling (whenever appropriate) and dimensionality reduction, as well as the final model fitting. Pipelines ensure that data processing takes place per training fold during cross validation, avoiding any data leakage to the whole training set.<br>
# 
# PCA is used for reducing the number of features. Principal components selection preserves 95% variance of the data, using Scikit's *'PCA'* method [PCA(n_components = 0,95)]. We do not select the components manually.<br>
# 
# 
# **Algorithm Selection and Final Evaluation**
# 
# KNN algorithm achieved the best CV score.
# Therefore, we train the model on the whole training set using RandomizedSearchCV. The optimal hyperparameter is *number of nearest neighbors = 4*.
# 
# References:
# [1]: Python Machine Learning - Third Edition by Vahid Mirjalili, Sebastian Raschka, Publisher: Packt Publishing, Release Date: December 2019

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().run_line_magic('pylab', 'inline')
#%pylab
import numpy as np
from sklearn import datasets, svm, metrics
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score,precision_score, f1_score, roc_curve, auc, make_scorer,roc_auc_score 
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
# import warnings filter
from warnings import simplefilter
simplefilter(action='ignore', category=DeprecationWarning)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")


# In[ ]:


X_train = train.loc[:,"pixel0":"pixel783"]
Y_train = train.loc[:, "label"]


# In[ ]:


X_test = test


# In[ ]:


RANDOM_STATE = 42
pca = PCA(n_components=0.95)
kfold = StratifiedKFold(n_splits=2, shuffle = True)
kfold


# In[ ]:


scoring = {'acc': 'accuracy'}


# ## KNN

# In[ ]:


#KNN
pipe_knn = Pipeline([('StdScaler', StandardScaler()),
                     ('minmax', MinMaxScaler()),
                     ('pca', pca),
                     ('clf_knn', KNeighborsClassifier())])


# KNN
rand_list_KNN = {'clf_knn__n_neighbors': sp_randint(1, 11),
                 'clf_knn__algorithm': ['auto']}

gs_KNN = RandomizedSearchCV(estimator = pipe_knn,
                    param_distributions = rand_list_KNN,
                    cv = kfold,
                    verbose = 2,
                    n_jobs = -1,
                    n_iter = 5)


scores_knn = cross_validate(gs_KNN, X_train, Y_train, cv = 5, scoring = scoring)

print('Cross Validation Scores:')
print('Accuracy: %.4f +/- %.4f' % (np.mean(scores_knn['test_acc']), np.std(scores_knn['test_acc'])))


# In[ ]:


scores_knn


# ## Naive Bayes

# In[ ]:


# Naive Bayes

pipe_nb = Pipeline([('pca', pca),
                    ('clf_nb', GaussianNB())])


rand_list_nb = {
                "clf_nb__var_smoothing": np.logspace(-9, 0, 5)
               }



gs_NB = RandomizedSearchCV(estimator = pipe_nb,
                    param_distributions = rand_list_nb,
                    cv = kfold,
                    verbose = 2,
                    n_jobs = -1,
                    n_iter = 5)



scores_nb = cross_validate(gs_NB, X_train, Y_train, cv = 5, scoring = scoring)

print('Cross Validation Scores:')
print('Accuracy: %.4f +/- %.4f' % (np.mean(scores_nb['test_acc']), np.std(scores_nb['test_acc'])))


# In[ ]:


scores_nb


# ## Random Forest

# In[ ]:


# random forest

pipe_rf = Pipeline([('pca', pca),
                    ('clf', RandomForestClassifier(n_estimators=100,
                      class_weight = 'balanced'))])


rand_list_rf = {"clf__max_depth": [5, None],
              "clf__max_features": sp_randint(1, 11),
              "clf__min_samples_split": sp_randint(2, 11),
              "clf__min_samples_leaf": sp_randint(1, 11),
              "clf__bootstrap": [True, False],
              "clf__criterion": ["gini", "entropy"]}



gs_RF = RandomizedSearchCV(estimator = pipe_rf,
                    param_distributions = rand_list_rf,
                    cv = kfold,
                    verbose = 2,
                    n_jobs = -1,
                    n_iter = 5)



scores_rf = cross_validate(gs_RF, X_train, Y_train, cv = 5, scoring = scoring)

print('Cross Validation Scores:')
print('Accuracy: %.4f +/- %.4f' % (np.mean(scores_rf['test_acc']), np.std(scores_rf['test_acc'])))


# In[ ]:


scores_rf


# ## AdaBoost Classifier

# In[ ]:


# AdaBoost
pipe_ada = Pipeline([('pca', pca),
                    ('clf_ada', AdaBoostClassifier())])


rand_list_ADA = {'clf_ada__learning_rate': np.logspace(-2, 2, 10),
                 'clf_ada__n_estimators': [100]
                 }

gs_ADA = RandomizedSearchCV(estimator = pipe_ada,
                    param_distributions = rand_list_ADA,
                    cv = kfold,
                    verbose = 2,
                    n_jobs = -1,
                    n_iter = 5)

scores_ada = cross_validate(gs_ADA, X_train, Y_train, cv = 5, scoring = scoring)

print('Cross Validation Scores:')
print('Accuracy: %.4f +/- %.4f' % (np.mean(scores_ada['test_acc']), np.std(scores_ada['test_acc'])))


# In[ ]:


scores_ada


# ## Winning algorithm: KNN

# In[ ]:


gs_KNN.fit(X_train.values, Y_train.values.ravel())
y_pred = gs_KNN.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


gs_KNN.best_params_


# ## Submission of predictions

# In[ ]:


submissions = pd.DataFrame({"ImageId": range(1,len(y_pred)+1), "Label": y_pred})
filename = 'MNIST_digits_predictions.csv'
submissions.to_csv(filename, index=False, header=True)
print('Saved file: ' + filename)


# In[ ]:


submissions.head

