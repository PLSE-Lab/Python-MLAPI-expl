#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Prediction
# 
# First of all, thank you for taking your time to read this notebook!
# 
# I have used the wine dataset to model the following Machine Learning algorithms for multinomial and binary clasification. 
# 
# * LogisticRegression
# * SVC (only for binary)
# * RandomForestClassifier
# * DesicionTreeClassifier
# * SGDClassifier (only for binary)
# * KNeighborsClassifier
# 
# Note that this notebok tries to show an introduction level for the specific concepts about Machine Learning.
# 
# Feel free to leave a message If you find any mistakes I have made.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
# import sklearn
import sklearn

import warnings
warnings.filterwarnings('ignore')

sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Get the data

# In[ ]:


wine = pd.read_csv('../input/winequality-red.csv')
wine.head()


# In[ ]:


wine.info()


# In[ ]:


wine.describe()


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

# split the data based on the wine quality
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(wine, wine["quality"]):
    strat_train_set = wine.loc[train_index]
    strat_test_set = wine.loc[test_index]


# In[ ]:


strat_train_set.shape


# In[ ]:


strat_test_set.shape


# ## Discover and Visualize the Data to Gain Insights

# In[ ]:


wine_strat = strat_train_set.copy()


# In[ ]:


wine_strat.quality.value_counts()


# Since we don't have enough data for each class, we coudn't expect a good accuracy for multi classification.

# In[ ]:


wine_strat.var()


# `total sulfur dioxide` has high variance!

# In[ ]:


wine_strat.corr()


# * the correlation between density and fixed acidity is moderate
# * the correlation between free sulfur dioxide and total sulfur dioxide is moderate

# In[ ]:


plt.figure(figsize=(10,5))


plt.scatter(x='free sulfur dioxide', y='total sulfur dioxide', c='quality', data=wine_strat)
plt.legend();


# In[ ]:


plt.figure(figsize=(10,5))

plt.scatter(x='density', y='fixed acidity', c='quality', data=wine_strat)
plt.legend();


# In[ ]:


from pandas.plotting import scatter_matrix

attributes = ["density", "fixed acidity", "free sulfur dioxide","total sulfur dioxide"]
scatter_matrix(wine_strat[attributes], figsize=(16, 10));


# In[ ]:


wine_strat.hist(figsize=(10,10));


# In[ ]:


wine_strat.alcohol.plot(kind='box',figsize=(10,10));


# In[ ]:


wine_strat['fixed acidity'].plot(kind='box',figsize=(10,10));


# In[ ]:


wine_strat['volatile acidity'].plot(kind='box',figsize=(10,10));


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine);


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine);


# In[ ]:


fig = plt.figure(figsize = (10,6))

sns.barplot(x = 'quality', y = 'fixed acidity', data = wine);


# In[ ]:


fig = plt.figure(figsize = (10,6))

wine_quality_density = wine.loc[:,['quality', 'density']]

wine_quality_density['density'] = np.log(wine_quality_density['density'])

sns.barplot(x = 'quality', y = 'density', data = wine_quality_density);


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine);


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine);


# In[ ]:


fig = plt.figure(figsize = (10,6))

wine_quality_ph = wine.loc[:,['quality', 'pH']].copy()

wine_quality_ph['pH'] = np.log(wine_quality_ph['pH'])

sns.barplot(x = 'quality', y = 'pH', data = wine_quality_ph);


# ## Prepare the Data

# In[ ]:


wine_train = strat_train_set.drop(columns=['quality'], axis=1).copy()
wine_train_labels = strat_train_set["quality"].copy()


# In[ ]:


wine_test = strat_test_set.drop(columns=['quality'], axis=1).copy()
wine_test_labels = strat_test_set["quality"].copy()


# In[ ]:


wine_train['total sulfur dioxide'] = np.log(wine_train['total sulfur dioxide'])
wine_test['total sulfur dioxide'] = np.log(wine_test['total sulfur dioxide'])

wine_train['pH'] = np.log(wine_train['pH'])
wine_test['pH'] = np.log(wine_test['pH'])

wine_train['density'] = np.log(wine_train['density'])
wine_test['density'] = np.log(wine_test['density'])


# In[ ]:


wine_train.var()


# In[ ]:


wine_train.head()


# ## Select and Train Models

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[ ]:


# create StandardScaler
st = StandardScaler()

wine_train_scaled = st.fit_transform(wine_train)
wine_test_scaled = st.fit_transform(wine_test)


# ### LogisticRegression

# In[ ]:


lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

lr.fit(wine_train_scaled, wine_train_labels)

lr.score(wine_train_scaled, wine_train_labels)


# In[ ]:


lr.score(wine_test_scaled, wine_test_labels)


# ### RandomForestClassifier

# In[ ]:


rfc = RandomForestClassifier(n_estimators=10)

rfc.fit(wine_train_scaled, wine_train_labels)

rfc.score(wine_train_scaled, wine_train_labels)


# In[ ]:


rfc.score(wine_test_scaled, wine_test_labels)


# ### DesicionTree

# In[ ]:


tree = DecisionTreeClassifier()

tree.fit(wine_train, wine_train_labels)

tree.score(wine_train, wine_train_labels)


# In[ ]:


tree.score(wine_test, wine_test_labels)


# In[ ]:


# feature importances
importances = tree.feature_importances_
importances


# In[ ]:


indices = np.argsort(importances)[::-1]

names = [wine_train.columns[i] for i in indices]
names


# ### KNeighborsClassifier

# In[ ]:


knc = KNeighborsClassifier()

knc.fit(wine_train_scaled, wine_train_labels)

knc.score(wine_train_scaled, wine_train_labels)


# In[ ]:


knc.score(wine_test_scaled, wine_test_labels)


# ## Binary Classification

# Inspired from https://www.kaggle.com/vishalyo990/prediction-of-quality-of-wine

# In[ ]:


# Bad = 0 and good = 1
wine_train_labels = np.where(wine_train_labels > 6.5, 1, 0)
wine_test_labels = np.where(wine_test_labels > 6.5, 1, 0)


# In[ ]:


wine_train_labels.sum()


# In[ ]:


wine_test_labels.sum()


# ###  LogisticRegression

# In[ ]:


lr = LogisticRegression(solver='lbfgs', max_iter=100)

lr.fit(wine_train_scaled, wine_train_labels)

lr.score(wine_train_scaled, wine_train_labels)


# In[ ]:


lr.score(wine_test_scaled, wine_test_labels)
# overfitting?


# In[ ]:


from sklearn.metrics import classification_report

y_pred_lr = lr.predict(wine_test_scaled)

print(classification_report(wine_test_labels, y_pred_lr, target_names=['bad', 'good']))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

accuracy_score(wine_test_labels, y_pred_lr), recall_score(wine_test_labels, y_pred_lr)


# ### SVC

# In[ ]:


svc = SVC()

svc.fit(wine_train_scaled, wine_train_labels)

svc.score(wine_train_scaled, wine_train_labels)


# In[ ]:


svc.score(wine_test_scaled, wine_test_labels)


# In[ ]:


y_pred_svc = svc.predict(wine_test_scaled)


# In[ ]:


print(classification_report(wine_test_labels, y_pred_svc, target_names=['bad', 'good']))


# ### LinearSVC

# In[ ]:


from sklearn.svm import LinearSVC

lsvc = LinearSVC(max_iter=1000)

lsvc.fit(wine_train_scaled, wine_train_labels)

lsvc.score(wine_train_scaled, wine_train_labels)


# In[ ]:


lsvc.score(wine_test_scaled, wine_test_labels)


# In[ ]:


y_pred_lsvc = lsvc.predict(wine_test_scaled)


# In[ ]:


print(classification_report(wine_test_labels, y_pred_lsvc, target_names=['bad', 'good']))


# ### RandomForestClassifier

# In[ ]:


rfc = RandomForestClassifier()

rfc.fit(wine_train_scaled, wine_train_labels)

rfc.score(wine_train_scaled, wine_train_labels)


# In[ ]:


rfc.score(wine_test_scaled, wine_test_labels)


# In[ ]:


y_pred_rfc = rfc.predict(wine_test_scaled)


# In[ ]:


print(classification_report(wine_test_labels, y_pred_rfc, target_names=['bad', 'good']))


# ### DecisionTreeClassifier

# In[ ]:


tree = DecisionTreeClassifier()

tree.fit(wine_train, wine_train_labels)

tree.score(wine_train, wine_train_labels)


# In[ ]:


tree.score(wine_test, wine_test_labels)


# In[ ]:


y_pred_tree = tree.predict(wine_test)


# In[ ]:


print(classification_report(wine_test_labels, y_pred_tree, target_names=['bad', 'good']))


# ### SGDClassifier

# In[ ]:


sgd = SGDClassifier(max_iter=1000, tol=1e-3)

sgd.fit(wine_train_scaled, wine_train_labels)

sgd.score(wine_train_scaled, wine_train_labels)


# In[ ]:


sgd.score(wine_test_scaled, wine_test_labels)


# In[ ]:


y_pred_sgd = sgd.predict(wine_test_scaled)


# In[ ]:


print(classification_report(wine_test_labels, y_pred_sgd, target_names=['bad', 'good']))


# ### KNeighborsClassifier

# In[ ]:


knc = KNeighborsClassifier()

knc.fit(wine_train_scaled, wine_train_labels)

knc.score(wine_train_scaled, wine_train_labels)


# In[ ]:


knc.score(wine_test_scaled, wine_test_labels)


# In[ ]:


y_pred_knc = knc.predict(wine_test_scaled)


# In[ ]:


print(classification_report(wine_test_labels, y_pred_knc, target_names=['bad', 'good']))


# ## Fine-Tune the Binary Models

# In[ ]:


pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])


# ###  SVC 

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid_svc = [
    {'classifier': [SVC()], 
     'classifier__kernel':['linear', 'rbf'],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
]


# In[ ]:


grid = GridSearchCV(pipe, param_grid_svc, scoring='accuracy', cv=10)


# In[ ]:


grid.fit(wine_train, wine_train_labels)


# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# In[ ]:


y_pred_grid_svc = grid.best_estimator_.predict(wine_test)

print(classification_report(wine_test_labels, y_pred_grid_svc, target_names=['bad', 'good']))


# In[ ]:


accuracy_score(wine_test_labels, y_pred_grid_svc)


# we have 90% accuracy on the test set using the best parameters.

# ###  RandomForestClassifier 

# In[ ]:


param_grid_rfc = [
    {'classifier': [RandomForestClassifier()], 
     'classifier__n_estimators': np.arange(2,20),
     'classifier__max_leaf_nodes' : np.arange(2,10),
     'classifier__max_depth': np.arange(2,10),
    }
]


# In[ ]:


grid_rfc = GridSearchCV(pipe, param_grid_rfc, scoring='accuracy', cv=10)


# In[ ]:


grid_rfc.fit(wine_train, wine_train_labels);


# In[ ]:


grid_rfc.best_params_


# In[ ]:


grid_rfc.best_score_


# In[ ]:


y_pred_grid_rfc = grid_rfc.best_estimator_.predict(wine_test)

print(classification_report(wine_test_labels, y_pred_grid_rfc, target_names=['bad', 'good']))


# In[ ]:


accuracy_score(wine_test_labels, y_pred_grid_rfc)


# we have 87% accuracy on the test set using the best parameters.

# ### DecisionTreeClassifier 

# In[ ]:


param_grid_dtc = [
    {'classifier': [DecisionTreeClassifier()], 
     'preprocessing': [None],  
     'classifier__criterion': ['gini', 'entropy'],
     'classifier__max_depth': np.arange(2,20)
    }
]


# In[ ]:


grid_dtc = GridSearchCV(pipe, param_grid_dtc, scoring='accuracy', cv=10)


# In[ ]:


grid_dtc.fit(wine_train, wine_train_labels);


# In[ ]:


grid_dtc.best_score_


# In[ ]:


grid_dtc.best_params_


# In[ ]:


y_pred_test_dtc = grid_dtc.best_estimator_.predict(wine_test)

print(classification_report(wine_test_labels, y_pred_grid_rfc, target_names=['bad', 'good']))


# In[ ]:


accuracy_score(wine_test_labels, y_pred_test_dtc)


# we have 91% accuracy on the test set using the best parameters.

# ## Conculusion
# 
# Thank you for reading and examining the notebook :)

# In[ ]:




