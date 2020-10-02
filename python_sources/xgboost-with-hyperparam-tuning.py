#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Machine learning competitions are a great way to improve your data science skills and measure your progress. 
# 
# In this notebook we will cover following steps:
# 1. Feature engineering
#     * Remove columns with missing values
#     * Numerical features
#     * Categorical features with low cardinality - one hot encoded
# 2. XGBoost model 
#     * Model validation using Training - CV using GridSearchCV
#     * Model hyper parameter tuning using GridSearchCV
#     * Model performance metrics: Mean absolute error
# 3. Model Insights
# 4. Submission for Kaggle competition

# ## 1. Feature Engineering

# In[ ]:


import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

# Train file path
train_file_path = '../input/train.csv'

home_data = pd.read_csv(train_file_path)
print("Original # of columns: {0}".format(len(home_data.columns)))

# Create y
y = home_data.SalePrice

# Candidate X
cols_with_missing = [col for col in home_data.columns 
                                 if home_data[col].isnull().any()]

X_candidate = home_data.drop(['Id','SalePrice'] + cols_with_missing, axis=1)

low_cardinality_cols = [cname for cname in X_candidate.columns if 
                                X_candidate[cname].nunique() < 10 and
                                X_candidate[cname].dtype == "object"]
numeric_cols = [cname for cname in X_candidate.columns if 
                                X_candidate[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

# One hot encoded
one_hot_encoded_X = pd.get_dummies(X_candidate[my_cols])
print("# of columns after one-hot encoding: {0}".format(len(one_hot_encoded_X.columns)))


# In[ ]:


# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file
test_data = pd.read_csv(test_data_path)
test_candidate = test_data.drop(['Id'] + cols_with_missing, axis=1)

# one hot encode
test_one_hot_encoded = pd.get_dummies(test_candidate[my_cols])
print("# of columns after one-hot encoding: {0}".format(len(test_one_hot_encoded.columns)))


# In[ ]:


X, test_X = one_hot_encoded_X.align(test_one_hot_encoded, join='left', axis=1)
print(np.shape(test_X))


# ## 2. XGBoost

# In[ ]:


# Cross Validation using GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 

xg_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb_pipeline = Pipeline([('xgb', xg_model)])

parameters = {'xgb__max_depth': range(2,5), 
              'xgb__min_child_weight': range(1,4)}

clf = GridSearchCV(xgb_pipeline, parameters,
                   scoring='neg_mean_absolute_error', cv=5)
clf.fit(X, y)


# In[ ]:


print("Best Score: {0}".format(clf.best_score_ * -1))
print(clf.best_estimator_)
print(clf.best_params_)


# ## 3. Model Insights

# In[ ]:


from matplotlib import pyplot as plt
import xgboost as xgb

fig, ax = plt.subplots(1,1,figsize=(10,10))
best_estimator_model = clf.best_estimator_.named_steps["xgb"]
xgb.plot_importance(best_estimator_model, max_num_features=15, ax=ax)


# In[ ]:


xgb.to_graphviz(best_estimator_model, num_trees=1)


# ## 4. Submission

# In[ ]:


print(np.shape(test_X))
#test_X = missing_data(test_X)

# make predictions which we will submit. 
test_preds = clf.predict(test_X)

# Submission format
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)


# In[ ]:




