#!/usr/bin/env python
# coding: utf-8

# This is intented to someone who has not implemented Lasso and Ridge regression at all.
# 
# 
# PS: There is no feature engineering, dimension reduction, cleaning data done in this code because it was mainly intended to practice Ridge and Lasso Regression.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_rand


# In[ ]:


boston_data = datasets.load_boston()
boston = pd.DataFrame(boston_data.data, columns = boston_data.feature_names)


# In[ ]:


boston['Target'] = boston_data.target


# In[ ]:


boston.head()


# In[ ]:


y = boston['Target']
x = boston.drop('Target',axis=1)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=22)


# In[ ]:


lr = LinearRegression()
lr = lr.fit(x_train,y_train)
lr_pred = lr.score(x_test,y_test)
lr_pred


# In[ ]:


rr = Ridge()
rr = rr.fit(x_train,y_train)
rr_pred =rr.score(x_test,y_test)
rr_pred


# In[ ]:


ls = Lasso()
ls =ls.fit(x_train, y_train)
ls_pred = ls.score(x_test,y_test)
ls_pred


# In[ ]:


alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
grid = GridSearchCV(estimator = rr,param_grid=dict(alpha=alphas))
grid.fit(x_train,y_train)
print(grid)
print("best score",grid.best_score_)
print('best estimator',grid.best_estimator_.alpha)


# In[ ]:


alphas = np.array([0.1,0.01,0.001,0.0001,0])
grid = GridSearchCV(estimator = ls,param_grid = dict(alpha=alphas))
grid.fit(x_train, y_train)
print(grid)
print("best score",grid.best_score_)
print("best estimator",grid.best_estimator_.alpha)


# In[ ]:


alphas = {'alpha': sp_rand()}
rsearch = RandomizedSearchCV(estimator=rr,param_distributions=alphas, n_iter=100)
rsearch.fit(x_train, y_train)
print(rsearch)
print("best score",rsearch.best_score_)
print("best estimator",rsearch.best_estimator_.alpha)


# In[ ]:


alphas = {'alpha': sp_rand()}
rsearch = RandomizedSearchCV(estimator=ls,param_distributions=alphas, n_iter=100)
rsearch.fit(x_train, y_train)
print(rsearch)
print("best score",rsearch.best_score_)
print("best estimator",rsearch.best_estimator_.alpha)

