#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# We wish to use GridsearchCV function from sklearn.model_selection module, so we wish to build our own model.
# That model should follow the BaseEstimator class from sklearn. So our first task is to build a model.
# The simples model is throwing random class from the set of classes. 
# 

# For building class we should implement to most important methods viz.
# 
# 1. Fit
# 2. score.

# In[ ]:


from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import accuracy_score

class random_class(BaseEstimator):
    def __init__(self, const=0, use_const=0):
        self.y = []
        self.cons = const
        self.use_cons = use_const
        
    def fit(self, X, y):
        self.y = y
        return self
    
    def score(self, X, y):
        np.asarray(X)
        n = np.shape(X)[0]
        
        ops = np.random.choice(self.y, n, replace=True) if self.use_cons else [self.cons]*n
        return np.random.choice([0.1, 0.2,0.5])


# ## Let's see how it works on the standard sklearn classifier

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


from sklearn.datasets import load_wine


# In[ ]:


data = load_wine()

X, y = data['data'], data['target']


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt = DecisionTreeClassifier()


# In[ ]:


params = {
    'max_depth': [4, 8],
    'max_features': [2, 4]
}

gsv = GridSearchCV(dt, params, cv=5).fit(X, y)

gsv.best_params_

gsv.verbose = 1

gsv.cv_results_


# ## Now using it on our own classifier

# In[ ]:


params = {'const': [1, 2], 
          'use_const': [0, 1]}

rdc = random_class()

gsv = GridSearchCV(rdc, params, cv=5).fit(X, y)


# In[ ]:


gsv.cv_results_


# In[ ]:


gsv.best_params_


# In[ ]:


gsv.best_score_


# That's it. If you want me to add more experiment. Comment.
