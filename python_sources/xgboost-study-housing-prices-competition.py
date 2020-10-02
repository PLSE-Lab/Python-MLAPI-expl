#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[6]:


# Load the train data
train_data = pd.read_csv("../input/train.csv")


# In[7]:


# Extract the target variable from the training data
y = train_data.SalePrice
train_data.drop('SalePrice', axis=1, inplace=True)


# In[8]:


# The final competition score is based on the MSE between the log of the test predictions and the log of the true SalePrice.
# With that in mind, always train to fit logy.
logy = np.log(y)


# In[9]:


# Separate the Id column from the predictive features
X = train_data.drop('Id', axis=1)


# In[13]:


X_numeric = X.select_dtypes(include=[np.number]).drop('MSSubClass', axis=1)
X_categorical = X.drop(X_numeric.columns, axis=1)

num_cols = X_numeric.columns
cat_cols = X_categorical.columns


# In[14]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ('num_imputer', SimpleImputer(strategy='median')),
    ('num_scaler', RobustScaler())
])

cat_pipeline = Pipeline([
    ('cat_nan_filler', SimpleImputer(strategy='constant', fill_value='not_in_data')),
    ('cat_onehot', OneHotEncoder(handle_unknown='ignore'))
])

minimal_preprocessor_pipeline = ColumnTransformer([
    ('num_pipeline', 'passthrough', num_cols),
    ('cat_pipeline', cat_pipeline, cat_cols)
])

preprocessor_pipeline = ColumnTransformer([
    ('num_pipeline', 'passthrough', num_cols),
    ('cat_pipeline', cat_pipeline, cat_cols)
])


# In[15]:


X = minimal_preprocessor_pipeline.fit_transform(X)


# In[17]:


from xgboost import XGBRegressor

xgb_regressor = XGBRegressor()


# In[18]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(xgb_regressor, X, logy, scoring='neg_mean_squared_error', cv=5, error_score=np.nan)


# Here I should highlight that this ran incredibly fast compared to any other model I've looked at.

# In[19]:


scores.mean()


# And even without any hyperparameter tuning, the results are competitive with any other model I've looked at.

# In[68]:


xgb_regressor = XGBRegressor(n_estimators=600)
scores = cross_val_score(xgb_regressor, X, logy, scoring='neg_mean_squared_error', cv=5, error_score=np.nan)


# In[70]:


scores.mean()
# Best: n_estimators=700, meanscore=-0.016198659607425607
# n_estimators >600, <800


# In[71]:


def gridsearch_fit_and_print_results(gridsearch, data, target):
    gridsearch.fit(data, target)
    
    print("Best parameters set found on development set:")
    print()
    print(gridsearch.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = gridsearch.cv_results_['mean_test_score']
    stds = gridsearch.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gridsearch.cv_results_['params']):
        print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))
    print()


# In[73]:


from sklearn.model_selection import GridSearchCV

# Defaults:
# max_depth=3, learning_rate=0.1, n_estimators=100
param_grid = {'max_depth': [2,3,4], 'learning_rate': [0.3, 0.1, 0.03], 'n_estimators': [300, 600, 900]}
xgb_regressor = XGBRegressor()
xgb_gridsearch = GridSearchCV(xgb_regressor, param_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)


# In[74]:


gridsearch_fit_and_print_results(xgb_gridsearch, X, logy)


# Best: -0.01562 (+/-0.00469) for {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 600}
# 
# Some interesting results elsewhere. I'll try expanding the parameter grid for max_depth=3.

# In[75]:


param_grid = {'max_depth': [3], 'learning_rate': [0.1, 0.07, 0.03], 'n_estimators': [600, 750, 900, 1050, 1200]}
xgb_regressor = XGBRegressor()
xgb_gridsearch = GridSearchCV(xgb_regressor, param_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)


# In[76]:


gridsearch_fit_and_print_results(xgb_gridsearch, X, logy)


# Best:
# 
# -0.01582 (+/-0.00648) for {'learning_rate': 0.07, 'max_depth': 3, 'n_estimators': 1200}
# 

# Going back to max_depth=2, learning_rate=0.1, and trying to find an optimal n_estimators.

# In[77]:


param_grid = {'max_depth': [2], 'learning_rate': [0.1], 'n_estimators': [400, 450, 500, 550, 600, 650, 700, 750, 800]}
xgb_regressor = XGBRegressor()
xgb_gridsearch = GridSearchCV(xgb_regressor, param_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)


# In[78]:


gridsearch_fit_and_print_results(xgb_gridsearch, X, logy)


# Best: -0.01561 (+/-0.00475) for {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 650}
# 

# In[79]:


param_grid = {'max_depth': [2], 'learning_rate': [0.1], 'n_estimators': [550, 575, 600, 625, 650, 675, 700]}
xgb_regressor = XGBRegressor()
xgb_gridsearch = GridSearchCV(xgb_regressor, param_grid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)


# In[80]:


gridsearch_fit_and_print_results(xgb_gridsearch, X, logy)


# Best: -0.01561 (+/-0.00475) for {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 650}
# 

# In[ ]:




