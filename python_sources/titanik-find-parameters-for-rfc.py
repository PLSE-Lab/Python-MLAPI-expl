#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from time import time


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
X_test = pd.read_csv('../input/titanic/test.csv')
id_for_subm = X_test['PassengerId'].copy()
train.head()


# In[ ]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# In[ ]:


X.info()


# In[ ]:


def preproc(train, test=[]):
    
    num_col = []
    cat_col = []
    cat_to_encode = []
    new_train = train.copy()
    new_test = test.copy() 
    
    for col in train.columns:
        if train[col].dtype == 'object': cat_col.append(col)
        else: num_col.append(col)
    
    num_imp = SimpleImputer(strategy='mean')
    new_train[num_col] = num_imp.fit_transform(new_train[num_col])
    new_test[num_col] = num_imp.transform(new_test[num_col])
    
    cat_imp = SimpleImputer(strategy='most_frequent')
    new_train[cat_col] = cat_imp.fit_transform(new_train[cat_col])
    new_test[cat_col] = cat_imp.transform(new_test[cat_col])
    
    for col in cat_col:
        a = new_train[col].unique()
        b = new_test[col].unique()
        if all(x in b for x in a): cat_to_encode.append(col)
        else: print('Drop', col)
    
    print('Numerical columns:', num_col)
    print('Categorical columns:', cat_to_encode)
    
    for col in cat_to_encode:
        cat_encoder = LabelEncoder()
        new_train[col] = cat_encoder.fit_transform(new_train[col])
        new_test[col] = cat_encoder.transform(new_test[col])
    
    col_to_keep = num_col+cat_to_encode

    return new_train[col_to_keep], new_test[col_to_keep]


# In[ ]:


X_train, X_test = preproc(X, X_test)


# In[ ]:


clf = RandomForestClassifier()
param_grid = {"max_depth": [7, 5, 3],
              "max_features": [3, 5, 7],
              "min_samples_split": [2, 3, 5],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [150, 200, 250, 300]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, iid=False)
start = time()
grid_search.fit(X_train, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))


# In[ ]:


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[ ]:


best = np.argmin(grid_search.cv_results_['rank_test_score'])
par = grid_search.cv_results_['params'][best]


# In[ ]:


report(grid_search.cv_results_)


# In[ ]:


eval_clf = RandomForestClassifier(**par)
eval_clf.fit(X_train, y)
pred = eval_clf.predict(X_test)


# In[ ]:


data_to_submit = pd.DataFrame({
    'PassengerId':id_for_subm,
    'Survived':pred
})
data_to_submit.to_csv('csv_to_submit.csv', index = False)

