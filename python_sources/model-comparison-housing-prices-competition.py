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


# In[ ]:


# read the data and store in DataFrame
train_data = pd.read_csv('../input/train.csv')


# ## Preprocessing: Train set only

# In[ ]:


# Extract the target variable
y = train_data.SalePrice
train_data.drop('SalePrice', axis=1, inplace=True)


# In[ ]:


# The final competition score is based on the MSE between the log of the test predictions and the log of the true SalePrice.
# With that in mind, always train to fit logy.
logy = np.log(y)
# Predictions will need to be of y, however, so for the final test submission, take the exponent of its output.


# ## Preprocessing

# In[ ]:


# Separate the Id column from the predictive features
X = train_data.drop('Id', axis=1)


# In[ ]:


# Compute a hash of each instance's identifier,
# keep only the last byte of the hash,
# and put the instance in the val set if the value of that byte is < val_ratio*256.
# (Hands-On Machine Learning with Scikit-Learn & TensorFlow, pg. 50 of my copy)

VAL_RATIO = 0.2
import hashlib
val_set_mask = train_data.Id.apply(lambda id : hashlib.md5(np.int64(id)).digest()[-1] < VAL_RATIO * 256)


# In[ ]:


# Separate a val set
val_X = X.loc[val_set_mask]
train_X = X.loc[~val_set_mask]
val_y = y.loc[val_set_mask]
train_y = y.loc[~val_set_mask]
val_logy = logy.loc[val_set_mask]
train_logy = logy.loc[~val_set_mask]


# In[ ]:


X_numeric = X.select_dtypes(include=[np.number]).drop('MSSubClass', axis=1)
X_categorical = X.drop(X_numeric.columns, axis=1)

num_cols = X_numeric.columns
cat_cols = X_categorical.columns


# In[ ]:


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

preprocessor_pipeline = ColumnTransformer([
    ('num_pipeline', num_pipeline, num_cols),
    ('cat_pipeline', cat_pipeline, cat_cols)
])


# In[ ]:


X = preprocessor_pipeline.fit_transform(X)
train_X = preprocessor_pipeline.transform(train_X)
val_X = preprocessor_pipeline.transform(val_X)


# ## Investigate some models

# In[ ]:


from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

N_ESTIMATORS = 100
rf_regressor = RandomForestRegressor(n_estimators=N_ESTIMATORS, criterion='mse')
et_regressor = ExtraTreesRegressor(n_estimators=N_ESTIMATORS, criterion='mse')


# Setting N_ESTIMATORS = 1000 did not improve validation error

# In[ ]:


rf_regressor.fit(train_X, train_logy)


# In[ ]:


from sklearn.metrics import mean_squared_error

rf_val_mse = mean_squared_error(rf_regressor.predict(val_X), val_logy)
rf_val_mse


# In[ ]:


et_regressor.fit(train_X, train_logy)


# In[ ]:


et_val_mse = mean_squared_error(et_regressor.predict(val_X), val_logy)
et_val_mse


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


rf_paramgrid = {'min_samples_leaf': [1,2,3,4,5],}
et_paramgrid = {'min_samples_leaf': [1,2,3,4,5],}

rf_gridsearch = GridSearchCV(rf_regressor, rf_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)
et_gridsearch = GridSearchCV(et_regressor, et_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)


# In[ ]:


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


# In[ ]:


gridsearch_fit_and_print_results(rf_gridsearch, X, logy)


# In[ ]:


gridsearch_fit_and_print_results(et_gridsearch, X, logy)


# There's only a slight difference between min_samples_leaf = 1 vs. 2. Both choices always do better than 3 or more. It's intuitively reasonable to expect that the model with min_samples_leaf = 2 will be more robust in general, compared to = 1.

# In[ ]:


rf_paramgrid = {'min_samples_leaf': [1,2], 'max_depth': [None, 10, 100], 'min_samples_split': [2, 3, 4, 5]}
et_paramgrid = {'min_samples_leaf': [1,2], 'max_depth': [None, 10, 100], 'min_samples_split': [2, 3, 4, 5]}

rf_gridsearch = GridSearchCV(rf_regressor, rf_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False, verbose=1, n_jobs=-1)
et_gridsearch = GridSearchCV(et_regressor, et_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False, verbose=1, n_jobs=-1)


# In[ ]:


gridsearch_fit_and_print_results(rf_gridsearch, X, logy)


# min_samples_leaf=1, min_samples_split=3, max_depth=None looks good

# In[ ]:


gridsearch_fit_and_print_results(et_gridsearch, X, logy)


# min_samples_leaf=2, min_samples_split anything <= 4, max_depth=None looks good

# In[ ]:


rf_paramgrid = {'min_samples_leaf': [1,2], 'max_depth': [None, 10, 100], 'min_samples_split': [2, 3, 4], 'max_features': ['sqrt']}
et_paramgrid = {'min_samples_leaf': [1,2], 'max_depth': [None, 10, 100], 'min_samples_split': [2, 3, 4], 'max_features': ['sqrt']}

rf_gridsearch = GridSearchCV(rf_regressor, rf_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False, verbose=1, n_jobs=-1)
et_gridsearch = GridSearchCV(et_regressor, et_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False, verbose=1, n_jobs=-1)


# In[ ]:


gridsearch_fit_and_print_results(rf_gridsearch, X, logy)


# None of these are as good as using all features, and none look like they're going to get better.

# In[ ]:


gridsearch_fit_and_print_results(et_gridsearch, X, logy)


# None of these are as good as using all features, and none look like they're going to get better.
# 

# In[ ]:


# Best models so far:
best_model_et = ExtraTreesRegressor(n_estimators=100, criterion='mse', min_samples_leaf=2)
best_model_rf = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_split=3)
# Mean validation score: -0.02010 (+/-0.00867) (changes randomly by a small amount from run to run)


# ## Linear Models
# Let's look at a simple linear model, and compare various regularization strategies.

# In[ ]:


# Ridge:
# " Minimizes the objective function:
# ||y - Xw||^2_2 + alpha * ||w||^2_2 "

# Lasso:
# " The optimization objective for Lasso is:
# (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1 "

# ElasticNet:
# "Minimizes the objective function:
# 1 / (2 * n_samples) * ||y - Xw||^2_2
# + alpha * l1_ratio * ||w||_1
# + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
# 
# l1_ratio = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable, unless you supply your own sequence of alpha."

from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge_regressor = Ridge(alpha=1.0, max_iter=10000) # default alpha value, 10x max_iter default (for sag solver)
lasso_regressor = Lasso(alpha=1.0, max_iter=10000) # default alpha value, 10x max_iter default (for sag solver)
en_regressor = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000) # default alpha and l1_ratio values, 10x max_iter default (for sag solver)


# In[ ]:


ridge_paramgrid = {'alpha': [1.0, 0.3, 0.1, 3.0, 10.0],}
lasso_paramgrid = {'alpha': [1.0, 0.3, 0.1, 3.0, 10.0],}
en_paramgrid = {'alpha': [1.0, 0.3, 0.1, 3.0, 10.0], 'l1_ratio': [0.5, 0.75, 0.25, 0.9, 0.1]}


# In[ ]:


ridge_gridsearch = GridSearchCV(ridge_regressor, ridge_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)
lasso_gridsearch = GridSearchCV(lasso_regressor, lasso_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)
en_gridsearch = GridSearchCV(en_regressor, en_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)


# In[ ]:


gridsearch_fit_and_print_results(ridge_gridsearch, X, logy)


# In[ ]:


gridsearch_fit_and_print_results(lasso_gridsearch, X, logy)


# 0.1 looks promising. Let's go smaller.

# In[ ]:


lasso_paramgrid = {'alpha': [0.1, 0.03, 0.01, 0.003, 0.001],}
lasso_gridsearch = GridSearchCV(lasso_regressor, lasso_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)
gridsearch_fit_and_print_results(lasso_gridsearch, X, logy)


# Go even smaller? Could potentially run into numerical issues with such a small alpha though.

# In[ ]:


lasso_paramgrid = {'alpha': [0.001, 0.0003, 0.0001, 0.00003, 0.00001],}
lasso_gridsearch = GridSearchCV(lasso_regressor, lasso_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)
gridsearch_fit_and_print_results(lasso_gridsearch, X, logy)


# Looks like the minimizer had trouble converging even with max_iter=10000. But based on the number of errors displayed, the first two alpha values look like they did converge, and already the trend is for MSE to go up when going from 0.001 to 0.0003. 
# 
# ExtraTrees regressor had a mean validation score of -0.02010 (+/-0.00867) for one run (it changes randomly from run to run, but not by a huge amount), while lasso regression with alpha=0.001 has a mean validation score of -0.01959 (+/-0.01687).

# In[ ]:


en_paramgrid = {'alpha': [0.0003, 0.001, 0.003], 'l1_ratio': [0.5, 0.75, 0.25, 0.9, 0.1]}
en_gridsearch = GridSearchCV(en_regressor, en_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)
gridsearch_fit_and_print_results(en_gridsearch, X, logy)


# ElasticNet regression with alpha=0.001, l1_ratio=0.5 has a mean validation score of -0.01907 (+/-0.01625), compared to lasso regression (alpha=0.001, l1_ratio=1.0) which has a mean validation score of -0.01959 (+/-0.01687).

# This alpha is so small that it may not be worth doing any regularization at all. Let's see what we get with ordinary least squares regression.
# 
# There aren't a lot of options for hyperparameters to tune, but as a sanity check we can see if fitting the intercept (i.e. allowing an overall bias) does a better job (which it should).

# In[ ]:


from sklearn.linear_model import LinearRegression

lin_regressor = LinearRegression()
lin_paramgrid = {'fit_intercept': [True, False]}
lin_gridsearch = GridSearchCV(lin_regressor, lin_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)
gridsearch_fit_and_print_results(lin_gridsearch, X, logy)


# So unregularized linear regression indeed does worse than mildly regularized linear regression.

# ## Best models found

# In[ ]:


best_models = [ElasticNet(alpha=0.001, l1_ratio=0.5),
               ExtraTreesRegressor(n_estimators=100, criterion='mse', min_samples_leaf=2),
              RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_split=3)]

