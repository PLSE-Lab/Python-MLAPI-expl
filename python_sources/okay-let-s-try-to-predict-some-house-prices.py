#!/usr/bin/env python
# coding: utf-8

# # First of all
# 
# Firstly, we've forked the Machine Learning Competitions notebook in order to check if we're able to achieve a better accuracy for our model, despite the bruteforcing process when using GridSearchCV. As before, we're creating and submiting predictions for a Kaggle competition, as per considered in the [Machine Learning Course / Housing Prices Competition](https://www.kaggle.com/learn/machine-learning).
# 
# Notes and inspirations:
# 
# https://www.kaggle.com/mrshih/here-s-a-house-salad-it-s-on-the-house
# 
# https://www.kaggle.com/carlolepelaars/eda-and-ensembling
# 
# https://www.kaggle.com/learn/machine-learning (The original learning track for this competition)

# ## Recap and reload!
# Here's the code we've written so far. Let's start by loading the libraries and databases, running it again and defining X, y and some other constants.
# 
# *Note that we'll use the cross validation process throughout this notebook although we could also use train_test_split*

# In[ ]:


# Code previously used to load data and some libraries

# The basic
import pandas as pd
import numpy as np

# Helpers
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline

# Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# Path of the file to read
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)


# path to file used for predictions
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)


# In[ ]:


# Creating target object
y = home_data.SalePrice

# Defining X
X_orig = home_data.drop('SalePrice', axis=1)

# Maintaining only numeric features over here and...
X_numeric_orig = X_orig.select_dtypes(exclude=['object'])

# ...Establishing the categoricals now!
X = pd.get_dummies(X_orig)

# Also, establishing the categoricals on the test data
test_X_categoricals = pd.get_dummies(test_data)

# creating test_X with only the numeric columns maintaining the previous pattern
test_X_numeric_orig = test_data.select_dtypes(exclude=['object'])


# Our *learning base* doesn't have the same features as our *predicting base*. Let's consider the same features, only, or our **predict** command will fail later.

# In[ ]:


# Listing features

XColumns = X.columns.tolist()
TestxColumns = test_X_categoricals.columns.tolist()


# In[ ]:


print('Learning base columns count: \n{}\n'.format(len(XColumns)))
print('Testing base columns count: \n{}\n'.format(len(TestxColumns)))


# In[ ]:


Features = []
Discard = []

# Note that the learning base have more columns than the testing base. Let's drop some of them.
for item in XColumns:
    if item in TestxColumns:
        Features.append(item)
    else:
        Discard.append(item)


# In[ ]:


# Redefining X
X = X[Features]


# In[ ]:


print("We've dropped the X columns on the learning data as below: \n")
print(Discard)


# In[ ]:


# Safety checking!

XColumns = X.columns.tolist()
TestxColumns = test_X_categoricals.columns.tolist()

print('New learning base columns count: \n{}\n'.format(len(XColumns)))
print('Testing base columns count: \n{}\n'.format(len(TestxColumns)))
print('.\n.\n.\n.\n\nGreat!!!')


# In[ ]:


# Imputing values

imputer = SimpleImputer()

X = pd.DataFrame(imputer.fit_transform(X))
test_X_categoricals = pd.DataFrame(imputer.fit_transform(test_X_categoricals))


# In[ ]:


# Aaaannnnd some other variables here
random_State_Seed = 1
scoring_type = 'neg_mean_absolute_error'


# In[ ]:


# Making scorer for the metric we'll use for the competition
def metric(val_predictions,val_y):
    return mean_absolute_error(val_predictions, val_y)

# Make scorer for scikit-learn
scorer = make_scorer(metric)


# In[ ]:


dectree_model = DecisionTreeRegressor()

# Grid for Decision Tree
dectree_grid = {
    'max_depth': [10, 20, 50, 80, 100],
    'random_state': [random_State_Seed],
    'max_leaf_nodes': [10, 20, 50, 80, 100, 500]
}


# In[ ]:


# Search parameter space
dectree_gridsearch = GridSearchCV(estimator = dectree_model, 
                                      param_grid = dectree_grid, 
                                      cv = 3, 
                                      n_jobs = -1, 
                                      verbose = 1, 
                                      scoring=scorer)


# In[ ]:


rf_model = RandomForestRegressor()

# Grid for Random Forest
rf_grid = {
    'n_estimators': [200, 500, 800, 1000],
    'random_state': [random_State_Seed],
    'max_leaf_nodes': [10, 20, 50, 80, 100, 500],
    
}


# In[ ]:


# Search parameter space
rf_gridsearch = GridSearchCV(estimator = rf_model, 
                                      param_grid = rf_grid, 
                                      cv = 3, 
                                      n_jobs = -1, 
                                      verbose = 1, 
                                      scoring=scorer)


# In[ ]:


xgb_model = XGBRegressor()

# XGB Grid
xgb_grid = {
    'learning_rate': [0.05, 0.07, 0.10], 
    'n_estimators': [800, 1000, 1500],
    'seed': [random_State_Seed],
}

# Split into validation and training data (XGB requires because of its evaluations)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=random_State_Seed)

evaluation_set = [(val_X, val_y)]

fit_params={"early_stopping_rounds": 5, 
            "eval_metric" : "mae", 
            "eval_set" : evaluation_set}


# In[ ]:


# Search parameters

xgb_gridsearch = GridSearchCV(estimator = xgb_model, 
                              param_grid = xgb_grid, 
                              #fit_params = fit_params, 
                              cv = 3,
                              n_jobs = -1, 
                              verbose = 1, 
                              scoring=scorer)


# In[ ]:


dectree_gridsearch.fit(X, y)


# In[ ]:


rf_gridsearch.fit(X, y)


# In[ ]:


xgb_gridsearch.fit(train_X, train_y, **fit_params) # Explainability about the **


# In[ ]:


# So...

print("So... What are the best parameters for each model?\n\n")

print('The best params for Decision Tree model:\n{}\n'.format(dectree_gridsearch.best_params_))
print('The best params for Random Forest model:\n{}\n'.format(rf_gridsearch.best_params_))
print('The best params for XBGBoost model:\n{}\n'.format(xgb_gridsearch.best_params_))


# In[ ]:


# And now, predicting...

print("What's the mae for each model?\n\n")

print('Error for Dec Tree:  {0:10.4f}'.format(metric(dectree_gridsearch.predict(X), y)))
print('Error for Random Forest: {0:10.4f}'.format(metric(rf_gridsearch.predict(X), y)))
print('Error for XGBoost:  {0:10.4f}'.format(metric(xgb_gridsearch.predict(X), y)))


# Now, let's predict using the best model over our testing base!

# In[ ]:


"""This is the original Submission cell. We've changed this one to the end of the notebook in order
to send the best model for the competition"""

# make predictions which we will submit. 
test_preds = xgb_gridsearch.predict(test_X_categoricals)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)


# # Testing the Work / Forking notebook
# After filling in the code above:
# 1. Click the **Commit and Run** button. 
# 2. After the code has finished running, click the small double brackets **<<** in the upper left of your screen.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 3. Go to the output tab at top of your screen. Select the button to submit your file to the competition.  
# 4. If you want to keep working to improve your model, select the edit button. Then you can change your model and repeat the process.
# 
