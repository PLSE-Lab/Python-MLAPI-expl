#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this notebook is to optimize all the models of the notebook: [Tactic 01. Test classifiers
# ](https://www.kaggle.com/juanmah/tactic-01-test-classifiers).
# 
# The models are fitted and predicted with the optimized parameters.
# The results are collected at [Tactic 99. Summary](https://www.kaggle.com/juanmah/tactic-99-summary).

# In[ ]:


import pip._internal as pip
pip.main(['install', '--upgrade', 'numpy==1.17.2'])
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels
from xgboost import XGBClassifier

import time
import pickle

from lwoku import get_prediction, add_features
from grid_search_utils import plot_grid_search, table_grid_search


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


# Add new features
print('     Number of features (before adding): {}'.format(len(X_train.columns)))
X_train = add_features(X_train)
X_test = add_features(X_test)
print('     Number of features (after adding): {}'.format(len(X_train.columns)))


# In[ ]:


print('  -- Drop features')
print('    -- Drop columns with few values (or almost)')
features_to_drop = ['Soil_Type7', 'Soil_Type8', 'Soil_Type15', 'Soil_Type27', 'Soil_Type37']
X_train.drop(features_to_drop, axis='columns', inplace=True)
X_test.drop(features_to_drop, axis='columns', inplace=True)
print('    -- Drop columns with low importance')
features_to_drop = ['Soil_Type5', 'Slope', 'Soil_Type39', 'Soil_Type18', 'Soil_Type20', 'Soil_Type35', 'Soil_Type11',
                    'Soil_Type31']
X_train.drop(features_to_drop, axis='columns', inplace=True)
X_test.drop(features_to_drop, axis='columns', inplace=True)
print('     Number of features (after drop): {}'.format(len(X_train.columns)))


# In[ ]:


features = X_train.columns.to_list()
with open("features.pickle", "wb") as fp:
    pickle.dump(features, fp)


# ## [Generalized Linear Models](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)

# # Logistic Regression classifier
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. LR](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-lr).

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-lr/clf_liblinear.pickle', 'rb') as fp:
    clf = pickle.load(fp)
lr_li_clf = clf.best_estimator_
lr_li_clf


# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-lr/clf_saga.pickle', 'rb') as fp:
    clf = pickle.load(fp)
lr_sa_clf = clf.best_estimator_
lr_sa_clf


# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-lr/clf_sag.pickle', 'rb') as fp:
    clf = pickle.load(fp)
lr_sg_clf = clf.best_estimator_
lr_sg_clf


# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-lr/clf_lbfgs.pickle', 'rb') as fp:
    clf = pickle.load(fp)
lr_lb_clf = clf.best_estimator_
lr_lb_clf


# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-lr/clf_newton-cg.pickle', 'rb') as fp:
    clf = pickle.load(fp)
lr_clf = clf.best_estimator_
lr_clf


# ## [Discriminant Analysis](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis)

# # Linear Discriminant Analysis
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. LDA](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-lda).

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-lda/clf.pickle', 'rb') as fp:
    clf = pickle.load(fp)
lda_clf = clf.best_estimator_
lda_clf


# ## [Nearest Neighbors](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)

# # k-nearest neighbors classifier
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. KNN](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-knn)

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-knn/clf.pickle', 'rb') as fp:
    clf = pickle.load(fp)
knn_clf = clf.best_estimator_
knn_clf


# ## [Naive Bayes](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)

# # Gaussian Naive Bayes
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. GNB](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-gnb)

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-gnb/clf.pickle', 'rb') as fp:
    clf = pickle.load(fp)
gnb_clf = clf.best_estimator_
gnb_clf


# ## [Support Vector Machines](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)

# # C-Support Vector Classification
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. SVC](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-svc)

# In[ ]:


svc_clf = SVC(random_state=42,
              verbose=True)


# ## [Ensemble Methods](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)

# 
# ### Bagging
# 

# # Bagging classifier
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. Bagging](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-bagging)

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-bagging/clf.pickle', 'rb') as fp:
    clf = pickle.load(fp)
bg_clf = clf.best_estimator_
bg_clf


# # Extra-trees classifier
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. Xtra-trees](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-xtra-trees)

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-xtra-trees/clf.pickle', 'rb') as fp:
    clf = pickle.load(fp)
xt_clf = clf.best_estimator_
xt_clf


# # Random forest classifier
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. RF](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-rf)

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-rf/clf.pickle', 'rb') as fp:
    clf = pickle.load(fp)
rf_clf = clf.best_estimator_
rf_clf


# ### Boosting

# # AdaBoost classifier
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. Adaboost](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-adaboost)

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-adaboost/clf.pickle', 'rb') as fp:
    clf = pickle.load(fp)
ab_clf = clf.best_estimator_
ab_clf


# # Gradient Boosting for classification
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. GB](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-gb)

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-gb/clf.pickle', 'rb') as fp:
    clf = pickle.load(fp)
gb_clf = clf.best_estimator_
gb_clf


# # LightGBM
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. LightGBM](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-lightgbm)

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-lightgbm/clf.pickle', 'rb') as fp:
    clf = pickle.load(fp)
lg_clf = clf.best_estimator_
lg_clf


# # XGBoost
# 
# Examine the full hyperparameter optimization details for this model in the following notebook: [Tactic 03. Hyperparameter optimization. XGBoost](https://www.kaggle.com/juanmah/tactic-03-hyperparameter-optimization-xgboost)

# In[ ]:


with open('../input/tactic-03-hyperparameter-optimization-xgboost/clf.pickle', 'rb') as fp:
    clf = pickle.load(fp)
xg_clf = clf.best_estimator_
xg_clf


# ### Model list

# In[ ]:


models = [
          ('lr_li', lr_li_clf),
          ('lr_sa', lr_sa_clf),
          ('lr_sg', lr_sg_clf),
          ('lr_lb', lr_lb_clf),
          ('lr', lr_clf),
          ('lda', lda_clf),
          ('knn', knn_clf),
          ('gnb', gnb_clf),
#           ('svc', svc_clf),
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


# # Compare

# In[ ]:


tactic_03_results = pd.read_csv('../input/tactic-03-hyperparameter-optimization/results.csv', index_col='Id', engine='python')
tactic_03_results


# In[ ]:


comparison = pd.DataFrame(columns = ['Model',
                                     'Accuracy',
                                     'Fit time',
                                     'Predict test set time',
                                     'Predict train set time'])

def get_increment(df1, df2, model, column):
    model1 = model.split('_', 1)[0]
    v1 = float(df1[df1['Model'] == model1][column])
    v2 = float(df2[df2['Model'] == model][column])
    return '{:.2%}'.format((v2 - v1) / v1)

for model in results['Model']:
    accuracy = get_increment(tactic_03_results, results, model, 'Accuracy')
    fit_time = get_increment(tactic_03_results, results, model, 'Fit time')
    predict_test_set_time = get_increment(tactic_03_results, results, model, 'Predict test set time')
    predict_train_set_time = get_increment(tactic_03_results, results, model, 'Predict train set time')
    comparison = comparison.append({
        'Model': model,
        'Accuracy': accuracy,
        'Fit time': fit_time,
        'Predict test set time': predict_test_set_time,
        'Predict train set time': predict_train_set_time
    }, ignore_index = True)    

comparison


# # Conclusions
# 

# There is a notable improve in the accuracy, except for knn.
