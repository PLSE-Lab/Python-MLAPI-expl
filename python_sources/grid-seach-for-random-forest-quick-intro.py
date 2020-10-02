#!/usr/bin/env python
# coding: utf-8

# # Grid Seach: Quick Intro

# Hello everyone. The idea of this notebook is to show how to do a gridsearch to find the best hyperparameters for our model. Making it clear that it can be any model, in this tutorial we use Random Forest because it is a simple but powerful model and because it makes sense to use a Forest to try to predict a forest. ;)

# First we import some libraries and we have some idea where the files are going to be used (thanks Kaggle!). Remembering that Numpy and Pandas are two important tools for working with data and the more familiar we are with it the simpler it will be to do what we came here to do.

# ## Our data

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


# After the libraries we bring the data to our environment.

# In[ ]:


train = pd.read_csv('/kaggle/input/learn-together/train.csv')
test = pd.read_csv('/kaggle/input/learn-together/test.csv')


# data_info is a very simple function defined to help us get an overview of our dataset. We have information about the data type, missings and amount of unique values. Only from this analysis can we get some very useful information that can help us select some features, avoid making noise in our model and even have some idea of new features to create ...
# 

# In[ ]:


def data_info(data):        
    info = pd.DataFrame()
    info['var'] = data.columns
    info['# missing'] = list(data.isnull().sum())
    info['% missing'] = info['# missing'] / data.shape[0]
    info['types'] = list(data.dtypes)
    info['unique values'] = list(len(data[var].unique()) for var in data.columns)
    
    return info


# In[ ]:


data_info(train)


# All of our features are numeric and there are no null values, in which case our challenge is much simpler and we can go straight to the model (this is just for this tutorial, in the real world we still have to do a lot of work before we get to the models).
# 
# First we separate our variables from our target:

# In[ ]:


x_train = train.drop(['Id', 'Cover_Type'], axis=1)
y_train = train['Cover_Type']


# ## Our model

# To make the selection of our model we will use cross validation, for this we first divide our data into k parts, here called folds, of the same size. After that, we trained a model in k-1 parts and validated on the data piece that was not used in training.

# In[ ]:


from sklearn.model_selection import KFold
cv_kfold = KFold(5, shuffle = False, random_state=12) 


# Here begins the magic. We define a function that will find the best hyperparameters for our model using grid search. To use this function we have six important parameters:
# * clf: the model we want to optimize, in this case Random Forest
# * X_train: our features
# * y_train: our target
# * params: (things get interesting here) params is a dictionary where keys are Random Forest parameters and values are various options for this parameter.
# * score: our evaluation metric, in this case accuracy
# * cv: cross validation, strategy to validate the model
# 

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

def grid_search(clf, X_train, y_train, params, score, cv):    
    grid = GridSearchCV(clf, params, scoring = score, cv = cv, return_train_score=True)
    grid_fitted = grid.fit(X_train, np.ravel(y_train))
    print ("Best score: %.4f" % grid_fitted.best_score_)
    print ("Best parameters: %s" % grid_fitted.best_params_)
    return grid_fitted, grid_fitted.best_estimator_, grid_fitted.cv_results_


# We define which parameters we want to try for our Random Forest. Remember that this is just an example and there are many other parameters that can be optimized and a multitude of different values that can be passed as an option for our search.

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.ensemble import RandomForestClassifier\n\nparams = {\n    'n_estimators':[128, 200, 256],\n    'criterion' : ['gini'],\n    'max_depth': [3, 5, 7, None],\n    'max_features': ['sqrt']\n}\nclf = RandomForestClassifier()\ngrid, model, results = grid_search(clf, x_train, y_train, params, 'accuracy', cv_kfold)\nmodel")


# After that we train our model with all the dataset we have available ...

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(x_train, y_train)')


# ... we make our prediction...

# In[ ]:


get_ipython().run_cell_magic('time', '', "x_test = test.drop('Id', axis=1)\ny_pred = model.predict(x_test)")


# ... and we make an amazing submission!

# In[ ]:


test['Cover_Type'] = y_pred
test[['Id', 'Cover_Type']].to_csv('submission.csv', index=False)


# Just curious to know how many of each class our model predicted :)

# In[ ]:


test.Cover_Type.value_counts()


# That's all folks, I hope this quick introduction can be helpful to you. If you have any questions please comment and we will all try to learn together!
