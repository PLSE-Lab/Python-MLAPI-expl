#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# In my [EDA](https://www.kaggle.com/franzxschmid/eda-for-beginners-bike-riding-data-set) for this dataset it was shown, that the bike-sharing-demand is characterized by knotty conditional and dependent relationships. 
# In such cases you shouldn't predict only with simple regressions. They can't explain that "knotty" part.
# You need a tree, better yet many trees or a forest. A relatively simple and effective technique for such cases is Random Forest Regression.

# **Data Preprocessing**

# Load libraries

# In[ ]:


import numpy as np
import pandas as pd


# Load dataset

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Adjust the train and the testset, in order to fit them together

# In[ ]:


test["casual"] = np.nan
test["registered"] = np.nan
test["count"] = np.nan

train["is_train"] = 1
test["is_train"] = 0


# Make a rowbind so that it's possible to clean the whole data at once

# In[ ]:


full = pd.concat([train, test], axis = 0)


# Correct the datatypes

# In[ ]:


full["datetime"] = full["datetime"].astype('datetime64')
full["season"] = full["season"].astype('category')
full["holiday"] = full["holiday"].astype('bool')
full["workingday"] = full["workingday"].astype('bool')
full["weather"] = full["weather"].astype('category')
full["is_train"] = full["is_train"].astype('bool')


# Get the variables "hour", "month" and "year" from the variable datetime

# In[ ]:


full["hour"] = full["datetime"].dt.hour.astype('category')
full["month"] = full["datetime"].dt.month.astype('category')
full["year"] = full["datetime"].dt.year.astype('category')
full = full.set_index('datetime')


# Feature selection
# * seasonal effects are already regarded through the variable "month"
# * "atemp" is highly correlated with "temp"
# * "registered" and "casual" are dependant variables and therefore not available in the  testdata

# In[ ]:


full = full.drop(["season", "atemp", "registered", "casual"], axis = 1)


# Get dummies for the factor variables

# In[ ]:


factor_variables = ["weather", "hour", "month", "year"]
full_dummies = pd.get_dummies(full[factor_variables])


# In[ ]:


non_factor_variables = ["holiday", "workingday", "temp", "humidity", "windspeed", "count", "is_train"]
full_no_dummies = full[non_factor_variables]


# In[ ]:


full = pd.concat([full_no_dummies, full_dummies], axis = 1)


# Split into train and testset

# In[ ]:


train = full[full["is_train"] == 1]
test = full[full["is_train"] == 0]


# In[ ]:


X_train = train.drop(["is_train", "count", "year_2012"], axis = 1)
y_train = train["count"]

X_test = test.drop(["is_train", "count", "year_2012"], axis = 1)


# ****Random Forest Regression and Hyperparameter-Tuning****
# 
# The hyperparameter-tuning will be down by hand in the following.  We can't test all combinations of parameters for computational reasons, therefore we are creating a rough parameter-grid and have a look in which area we find the "best" parameters. Than we investigate this area with a smaller but finer grid. We are doing this several times iteratively.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# Instantiate a Random Forest Regressor

# In[ ]:


rfr = RandomForestRegressor()


# Create a first rough hyperparameter-grid

# In[ ]:


param_grid = [
  {'max_depth': [20, 30, 40, 50],
   'min_samples_leaf': [1, 5, 10, 20],
   'n_estimators':[30, 50, 70, 90],
   'random_state':[42]}]


# Fit the Random Forest Regressor with a cross validated gridsearch

# In[ ]:


rfr = GridSearchCV(rfr, param_grid, cv = 5, scoring = 'neg_mean_squared_log_error')
rfr.fit(X_train, y_train)


# Print the best score

# In[ ]:


rfr.best_score_


# From the best parameters we can conclude that
# * the "best" maximal depth is near 30
# * the "best" minimal samples leaf is beetween 1 and 5
# * the "ideal" number of estimators is near 50

# In[ ]:


rfr.best_params_


# Now we investigate, if there are even better parameters near the values we got from the last output

# In[ ]:


rfr = RandomForestRegressor()
param_grid = [
  {'max_depth': [25, 30, 35],
   'min_samples_leaf': [1, 3, 5],
   'n_estimators':[40, 50, 60],
   'random_state':[42]}]
rfr = GridSearchCV(rfr, param_grid, cv = 5, scoring = 'neg_mean_squared_log_error')
rfr.fit(X_train, y_train)


# The score gets a bit better

# In[ ]:


rfr.best_score_


# Now we conclude
# * the "best" maximal depth is very close to 30
# * the "best" minimal samples leaf can only be 1 or 2
# * the "ideal" number of estimators seems to be closer to 60 and not to 50

# In[ ]:


rfr.best_params_


# In[ ]:


rfr = RandomForestRegressor()
param_grid = [
  {'max_depth': [28, 30, 32],
   'min_samples_leaf': [1, 2],
   'n_estimators':[55, 60, 65],
   'random_state':[42]}]
rfr = GridSearchCV(rfr, param_grid, cv = 5, scoring = 'neg_mean_squared_log_error')
rfr.fit(X_train, y_train)


# The score still gets a bit better

# In[ ]:


rfr.best_score_


# We conclude
# * the "best" maximal depth can only be 29, 30 or 31
# * the "best" minimal samples leaf is 1 or 
# * the "ideal" number of estimators is close to 55

# In[ ]:


rfr.best_params_


# In[ ]:


rfr = RandomForestRegressor()
param_grid = [
  {'max_depth': [29, 30, 31],
   'min_samples_leaf': [1],
   'n_estimators':[50, 55, 60],
   'random_state':[42]}]
rfr = GridSearchCV(rfr, param_grid, cv = 5, scoring = 'neg_mean_squared_log_error')
rfr.fit(X_train, y_train)


# The score and the "best" parameters are the same as before

# In[ ]:


rfr.best_score_


# In[ ]:


rfr.best_params_


# Conclusion
# * the "best" maximal depth is 30 
# * we can only refine our hyperparameters if we find a better number of estimators than 55

# In[ ]:


rfr = RandomForestRegressor()
param_grid = [
  {'max_depth': [30],
   'min_samples_leaf': [1],
   'n_estimators':[52, 53, 54, 55, 56, 57, 58],
   'random_state':[42]}]
rfr = GridSearchCV(rfr, param_grid, cv = 5, scoring = 'neg_mean_squared_log_error')
rfr.fit(X_train, y_train)


# It's still the same score because 55 was already the "best" number of estimators

# In[ ]:


rfr.best_score_


# In[ ]:


rfr.best_params_


# In[ ]:


prediction = rfr.predict(X_test)
result = pd.DataFrame(test.index).assign(count = prediction)
result.to_csv('output_pred.csv', index=False)

