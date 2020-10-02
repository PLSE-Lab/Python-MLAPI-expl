#!/usr/bin/env python
# coding: utf-8

# **<span style="color:blue; font-size:1.2em;">Update 1, July 12th, 2020: "Drop code, improve ranking."</span>** This notebook shows how far you can go just by applying XGBoost with deliberate/handwritten hyperparameter tuning and minimal data preprocessing, in particular w/o any feature engineering. As in the exercises, when chosing categorical columns, I only kept "low-cardinality" (<= 10 distinct values) features in the first version. But just like XGBoost can deal with missing values very well, it is very good at processing sparse data that is introduced by one-hot encoding categorical data. Enough said - what I did is this: Just dropping the "if"-clause that restricted categorical columns to <= 10 distinct values and thereby simply keeping *all* categorical features (still w/o further engineering) and what happened: I gained a few hundred places in the leaderboard now reaching the Top 2% with a MAE of 13099.
# Another minor update: In the last optimization step (number of estimators by "early stopping") I increased `early_stopping_rounds` from 50 to 100.
# 
# # Hi all!
# In this notebook I share some of my thoughts after completing the Intermediate Machine Learning course and the approach I finally came up with to significantly improve my scoring in the House Pricing competition. 
# When I completed the Intermediate ML course I felt like I need to wrap-up and sort things out before proceeding to the next course. That was for two reasons: First, it did not become very apparent in my opinion how the last few exercises (Pipelines, Cross-Validation, XGB, Data Leakage) fit together. Second, the last exercises did not require submissions to the House Pricing competition and hence my scoring was not very satisfying.
# I wondered how I could combine the learned techniques and improve my scoring. It took me some time, some extra research and some trial and error, but in the end I managed to train a model that reached a MAE score of around 13590 (Top 3%) by only applying techniques that were covered or mentioned in the Intermediate ML course. In particular, I did not do any feature engineering. The code is completely listed. You can run it to reproduce the result. 
# 
# *To summarize, I chose XGBoost as model and developed a strategy to tune some (hyper-)parameters by repeated cross validations. Two design decisions were i) to mostly avoid the XGBoost "early stopping" feature in favor of using XGBoost within scikit-learn's pipelines and grid search and ii) to sequentially apply several rounds of parameter searches in order to avoid a combinatorial explosion when testing the possible combinations of all parameters together.*
# 
# Any questions and feedback in the comments are very welcome and if you liked the material I would be happy about an upvote.

# # Import packages and load data
# This is basically copied from the exercises.

# In[ ]:


import pandas as pd
#from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Read the data
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]

# Update 1: chose all categorical columns instead of only low cardinality features (see above)
categorical_cols = [cname for cname in train_data.columns if train_data[cname].dtype == "object"]
# low_cardinality_cols = [cname for cname in train_data.columns if train_data[cname].nunique() < 10 
#                        and train_data[cname].dtype == "object"]

# Keep selected cols
my_cols = categorical_cols + numeric_cols
X = train_data[my_cols].copy()
X_test = test_data[my_cols].copy()

# We use cross validation and hence do not need to split into train and validation set
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
#                                                               random_state=0)


# # Setting-up the pipeline
# In order to use XGBoost we need to encode categorical data. I took one-hot encoding since we learned that it "typically performs best". It is not necessary to impute missing numerical data since this is done by XGBoost automatically. Of course it is possible to impute missing numerical data as a data preprocessing step but in a quick comparison it turned out that keeping missing numerical values unchanged achieves better results. It is important, however, to explicitly tell the pipeline that numerical data shall be kept - otherwise numerical columns will completely be dropped and that - of course - results in very bad scorings. This is done by using the "passthrough" transformer.
# Regarding the learning rate for XGBoost we learned that a small rate generally produces more accurate models. In the XGBoost lesson the learning rate was set to 0.05. I ran a few experiments with an even smaller value of 0.01 and observed that this lower learning rate results in better scores.

# In[ ]:


# Preprocessing for numerical data
# As mentioned above, we don't impute missing numerical data, 
# so this is out-commented
#numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),   # !!! important to keep numeric data
        ('cat', categorical_transformer, categorical_cols)
    ])

# instantiating the XGBoost model and the pipeline
model = XGBRegressor(learning_rate=0.01, random_state=0)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


# # Parameter tuning
# In addition to `learning_rate` that we fix to 0.01, the XGBoost lesson covered `n_estimators`, i.e. the number of used estimators / decision trees as another important XGBoost parameter. There are several other parameters that affect model accuracy - mostly by controlling possible overfitting. In addition to `n_estimators` we are going to tune
# * the maximal depth (`max_depth`, default 6) of the learned decision trees, 
# * the minimum number of samples (`min_child_weight`, default 1) required in order to create a further node in the tree, 
# * the fraction of training data / rows (`subsample`, default 1) that is used in each training step, and 
# * the fraction of features / columns (`colsample_bytree`, default 1) that is used for each tree.
# 
# (Read more in the [XGBoost documentation](https://xgboost.readthedocs.io/en/latest/parameter.html)).
# 
# In the XGBoost lesson we learned about this handy "early stopping" feature of XGBoost to avoid overfitting by stopping to add further estimators when the validation score stops improving. It requires, however, to provide a validation set and this does not fit well with using pipelines and even worse with cross validation were validation sets are not provided explicitly but computed automatically as part of the cross validation technique. Therefore, I decided to mostly avoid using this feature and instead to handle the number of estimators just like the other parameters. 
# 
# My goal was to tune the mentioned parameters within the following inclusive ranges:
# * `n_estimators`: 200 to 2000 
# * `max_depth` and `min_child_weight`: 1 to 8
# * `subsample` and `colsample_bytree`: 0.5 to 1
# 
# If we wanted to try every combination (`n_estimators`, `max_depth`, `min_child_weight` in steps of 1; `subsample`, `colsample_bytree` in steps of 0.1) and used cross validation with 5 folds, we would have 1800 x 8 x 8 x 6 x 6 x 5 training episodes. If one episode took one second, **the complete tuning would take 240 days** -- not practicable!
# 
# My approach here is to i) not tune all parameters together but sequentially and ii) for each parameter to first apply a coarse search to find the best "region" and then fine tune by only search within that region. The order of adding parameters to the search is inspired by [this blog post](https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f). 
# To summarize, my tuning scheme works as follows:
# 1. Find the optimal number of estimators with 200 to 2000 in steps of 200. 
# 2. Find the best combination of `max_depth` and `min_child_weight` with possible values 1, 4, 7 each.
# 3. Fine tune `max_depth` and `min_child_weight` by also testing the values 1 smaller and greater than the current best found values.
# 4. Find the best combination of `subsample` and `colsample_bytree` with possible values 0.6 and 0.9 each.
# 5. Fine tune `subsample` and `colsample_bytree` by also testing values 0.1 smaller and greater than the current best found values.
# 6. Find the exact optimal number of estimators with the "early stopping" feature.
# 
# By applying this scheme, the complete tuning is finished in less than half an hour. The prize is, of course, that we exclude a very large fraction of possible parameter combinations and very likely also the best combination. The hope and expectation however is that we include "sufficiently good" combinations from which we finally find the best one.
# 
# The search in the parameter space (the possible parameter values in each step) is done by using the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) function from scikit-learn mentioned in the cross validation lesson. It simply enumerates all combinations of provided parameter values, cross validates each one, and stores the combination with the best score.
# 
# In order to pass parameter values to one particular pipeline step (the "model" step in this case), we need to prefix the parameter names with the step name, e.g. `model__n_estimators`.
# 
# In the following code, let the print statements uncommented to observe the evolution of the parameter space and how the corresponding cross validation scores increase from step to step.

# In[ ]:


# The initial parameter search space. 
# We coarsely search for the best number of estimators within 200 to 2000 in steps of 200.
parameter_space = {
    'model__n_estimators': [n for n in range(200, 2001, 200)]
}
print("Initial parameter search space: ", parameter_space)

# Initializing the grid search.
folds = KFold(n_splits=5, shuffle=True, random_state=0)
grid_search = GridSearchCV(my_pipeline, param_grid=parameter_space, 
                           scoring='neg_mean_absolute_error', cv=folds)

# First search round.
grid_search.fit(X, y)
print("Best found parameter values: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Fix n_estimators to the best found value
parameter_space['model__n_estimators'] = [grid_search.best_params_['model__n_estimators']]

# We add max_depth and min_child_weight with possible values 1, 4, 7 each to the search.
parameter_space['model__max_depth'] = [x for x in [1, 4, 7]]
parameter_space['model__min_child_weight'] = [x for x in [1, 4, 7]]
print("Updated parameter space: ", parameter_space)

# Search.
grid_search.fit(X, y)
print("Best found parameter values: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Fine tuning.
fine_tune_range = [-1, 0, 1]
# Update the parameter space for fine tuning.
parameter_space['model__max_depth'] = [grid_search.best_params_['model__max_depth'] + i 
                                       for i in fine_tune_range]
parameter_space['model__min_child_weight'] = [grid_search.best_params_['model__min_child_weight'] + i 
                                              for i in fine_tune_range]
print("Updated parameter space: ", parameter_space)

# Search.
grid_search.fit(X, y)
print("Best found parameter values: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# We now have fixed the final values for max_depth and min_child_weight 
# and update the search space accordingly.
parameter_space['model__max_depth'] = [grid_search.best_params_['model__max_depth']]
parameter_space['model__min_child_weight'] = [grid_search.best_params_['model__min_child_weight']]

# Add subsample and colsample_bytree with possible values 0.6 and 0.9 each.
parameter_space['model__subsample'] = [x/10 for x in [6, 9]]
parameter_space['model__colsample_bytree'] = [x/10 for x in [6, 9]]
print("Updated parameter space: ", parameter_space)

# Search.
grid_search.fit(X, y)
print("Best found parameter values: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Fine tuning for subsample and colsample_bytree and fixing their final values
parameter_space['model__subsample'] = [grid_search.best_params_['model__subsample'] + i/10 
                                       for i in fine_tune_range]
parameter_space['model__colsample_bytree'] = [grid_search.best_params_['model__colsample_bytree'] + i/10 
                                              for i in fine_tune_range]
print("Updated parameter space: ", parameter_space)

grid_search.fit(X, y)

parameter_space['model__subsample'] = [grid_search.best_params_['model__subsample']]
parameter_space['model__colsample_bytree'] = [grid_search.best_params_['model__colsample_bytree']]

# Parameter values so far...
print("Fixed parameter values: ", parameter_space)


# We now have fixed all parameter values but searched only coarsely for the best number of estimators. In a final optimization step we use XGBoost's early-stopping feature to fine-tune it. We use XGBoosts plain python API in order to apply XGBoost's internal cross validation function.

# In[ ]:


import xgboost as xgb
X_enc = pd.get_dummies(X)
dtrain = xgb.DMatrix(X_enc, label=y)

# Setting up parameter dict with found optimal values
params = {
    'max_depth': parameter_space['model__max_depth'][0],
    'min_child_weight': parameter_space['model__min_child_weight'][0],
    'eta': 0.01, # eta is the learning rate
    'subsample': parameter_space['model__subsample'][0],
    'colsample_bytree': parameter_space['model__colsample_bytree'][0]
}
print(params)
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=10000,
    seed=0,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=100
)
#print(cv_results)

mae = cv_results['test-mae-mean'].min()
opt_n_estimators = cv_results['test-mae-mean'].argmin()

print("Optimal number of estimators: ", opt_n_estimators)
print("Score: ", mae)


# # Final training and test predictions
# All parameters are fixed and we train the final model.

# In[ ]:


params = {
    'n_estimators': opt_n_estimators,
    'learning_rate': 0.01, 
    'max_depth': parameter_space['model__max_depth'][0],
    'min_child_weight': parameter_space['model__min_child_weight'][0],
    'subsample': parameter_space['model__subsample'][0],
    'colsample_bytree': parameter_space['model__colsample_bytree'][0]
}

final_model = XGBRegressor(**params, random_state=0)
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', final_model)])

#print(final_pipeline.named_steps['model'])

final_pipeline.fit(X, y)


# Finally we compute predictions on the test data set and save them.

# In[ ]:


# Compute predictions on test data and save to file.
preds_test = final_pipeline.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

