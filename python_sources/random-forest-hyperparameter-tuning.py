#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# reading train and test dataset

# In[ ]:


train = pd.read_csv('/kaggle/input/predict-who-is-more-influential-in-a-social-network/train.csv')
test = pd.read_csv('/kaggle/input/predict-who-is-more-influential-in-a-social-network/test.csv')
submission = pd.read_csv('/kaggle/input/predict-who-is-more-influential-in-a-social-network/sample_predictions.csv')
train.head(5)


# shape of train and test data

# In[ ]:


print(train.shape)
print(test.shape)
print((train.columns).difference(test.columns))


# Number of null values in train and test data

# In[ ]:


train.isnull().sum().sum()
test.isnull().sum().sum()


# checking whether dataset is balanced or not

# In[ ]:


train.Choice.value_counts()


# In[ ]:


y = train['Choice']
x = train.drop('Choice',axis=1)


# Scaling is not required
# 
# Scaling is done to Normalize data so that priority is not given to a particular feature. Role of Scaling is mostly important in algorithms that are distance based and require Euclidean Distance.
# 
# Random Forest is a tree-based model and hence does not require feature scaling.
# 
# This algorithm requires partitioning, even if you apply Normalization then also> the result would be the same.

# splitting data in training and test data

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# Randomforestclassifier with no hyperparameter tuning

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_split=5, min_samples_leaf=2, max_features='auto', bootstrap=True, n_jobs=-1, random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# RandomTreeClassifier hyperparameter tuning by RandomizedSearchCV

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in range(200,2000,200)]
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
print(random_grid)


# Training parameter grid by RandomizedSearchCV

# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, y_train)


# Best parameters

# In[ ]:


rf_random.best_params_


# model accuracy

# In[ ]:


from sklearn import metrics

def evaluate(model, test_features, test_labels):
    y_pred = model.predict(test_features)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print (accuracy)
    print(confusion_matrix(y_test,y_pred))
    

best_random = rf_random.best_estimator_
evaluate(best_random, x_test, y_test)


# hyperparameter tuning by GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10,15],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5,6],
    'min_samples_split': [3,4,5,6],
    'n_estimators': [1150, 1200, 1250, 1300,1350]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(x_train,y_train)


# best parameter from gridseachCV

# In[ ]:


grid_search.best_params_


# Performance of best parameter

# In[ ]:


best_grid = grid_search.best_estimator_
evaluate(best_grid,x_test,y_test)


# best model prediction on test data

# In[ ]:


sub = best_random.predict(test)
sub


# In[ ]:


submission['Choice'] = sub


# submission file

# In[ ]:


submission.to_csv('submission2.csv',index=False)

