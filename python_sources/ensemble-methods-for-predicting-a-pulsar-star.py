#!/usr/bin/env python
# coding: utf-8

# <h1>Predicting a Pulsar Star</h1>
# The purpouse here is to use multiple ensemble methods to perform a classification and mark result as either pulsar star or RFI/noise. I will start from decision tree classifier and then will use a couple of ensemble methods based on decision trees. Different quality metrics for those methods will be compared. 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import graphviz 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Firstly, let's look at the data structure. From data description we know that target classes are balanced: the data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real pulsar examples. There are no categorical values. Also as we use decision trees and algorithms based on them we can skip normalization part.

# In[2]:


df = pd.read_csv('../input/pulsar_stars.csv')
df.head()


# Luckily there are no missing values so we can proceed to classification.

# In[3]:


df.info()


# Let's extract target class and split dataset to train (75%) and test (25%) samples.

# In[4]:


y = df['target_class']
del df['target_class']
x = df
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[11]:


def print_scores(scores, classifier):
    print('---------------------------------')
    print(str(classifier))
    print('--------------')
    print('test score mean', scores['test_score'].mean())
    print('test score std', scores['test_score'].std())
    print('train score mean', scores['train_score'].mean())
    print('train score std', scores['train_score'].std())
    print('-----------------------------------')
    
def run_classification(clf):
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)
    scores = cross_validate(clf, x_train, y_train, return_train_score=True, cv=5)
    print_scores(scores, clf)
    print(classification_report(y_test, y_pred))
    print('----------------------------------')
    print('ROC AUC score', roc_auc_score(y_test, y_pred_proba[:, 1]))


# <h4>Decision tree</h4>
# I will start with simple decision tree classifier. Below are the results of multiple metrics. Cross-validation results are listed, as well as sklearn classification report. If we look at ROC AUC metric we see the score 0.9.

# In[20]:


dt_clf = DecisionTreeClassifier(random_state=0)
run_classification(dt_clf)


# <h4>Random forest</h4>
# Next classifier will be Random Forest. As it takes `n_estimators` parameter, which specifies number of trees, we can vary its values and choose the best one.

# In[8]:


grid_clf = GridSearchCV(RandomForestClassifier(random_state=0),
                        param_grid={'n_estimators': [100, 500, 1000]}, cv=5)
grid_clf.fit(x_train, y_train)
print('Best param', grid_clf.best_params_)


# The best one appeared to be 500, so we run classification process with 500 trees. ROC AUC value (0.975) as well as other metrics is better than for decision tree.

# In[9]:


run_classification(RandomForestClassifier(n_estimators=500, random_state=0))


# <h4>AdaBoost</h4>
# The next classifier is AdaBoost. Here we find out max number of weak learners that can be used. It appears to be 500.

# In[10]:


grid_ada = GridSearchCV(AdaBoostClassifier(random_state=0),
                        param_grid={'n_estimators': [100, 500, 1000]}, cv=5)
grid_ada.fit(x_train, y_train)
print('Best param', grid_ada.best_params_)


# The results of classification are slightly worse than for random forest.

# In[21]:


run_classification(AdaBoostClassifier(n_estimators=500, random_state=0))


# <h4>Gradient Boosting</h4>
# Now let's move to gradient boosting and its' parameters tuning. 100 is the best value for `n_estimators`.

# In[23]:


grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=0),
                       param_grid={'n_estimators': [100, 500, 1000]}, cv=5)
grid_gb.fit(x_train, y_train)
print('Best param', grid_gb.best_params_)


# With ROC AUC score around 0.978, results for gradient boosting are slightly better than for random forest.

# In[24]:


run_classification(GradientBoostingClassifier(n_estimators=100, random_state=0))


# <h4>XGBoost</h4>
# The last one will be XGBoost classifier. The best `n_estimators` parameter is 100, so the default one.

# In[25]:


grid_xg = GridSearchCV(XGBClassifier(random_state=0), param_grid={'n_estimators': [100, 500, 1000]}, cv=5)
grid_xg.fit(x_train, y_train)
print('Best param', grid_xg.best_params_)


# The results of classification are the best among classifiers that were used here, so XGBoost becomes the winner!

# In[26]:


run_classification(XGBClassifier(random_state=0))


# In[ ]:




