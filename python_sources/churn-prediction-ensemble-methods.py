#!/usr/bin/env python
# coding: utf-8

# <h2>Introduction</h2>
# 
# In this notebook we will be using ensemble methods to predict if a customer left the company in the last month. The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve robustness. There are two general categories of ensemble methods: averaging and boosting methods. Decision trees will be used as our base estimator for both.

# In[ ]:


import random
import pprint
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head(3)


# <h2>1. Preprocessing</h2>
# 
# Since all ensemble methods in this notebook are based on Decision Trees, we don't need feature scalling. However, we need to encode categorical features and drop a few columns. 

# In[ ]:


def preprocessing(df):
    """Preprocess df and return X (train features) and Y (target feature)."""
    # Impute missing values on TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(" ", 0).astype('float32')
    # Drop customer id
    df.drop(['customerID'], axis=1, inplace=True)
    # Encode categorical features
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' or c == 'SeniorCitizen']
    for col in cat_cols:
        if df[col].nunique() == 2:
            df[col], _ = pd.factorize(df[col])
        else:
            df = pd.get_dummies(df, columns=[col])
    # Drop target column and some correlated features
    drop_features = ['OnlineSecurity_No internet service', 'OnlineBackup_No internet service',
                     'DeviceProtection_No internet service', 'TechSupport_No internet service',
                     'StreamingTV_No internet service', 'StreamingMovies_No internet service',
                     'PhoneService', 'Churn']
    feats = [c for c in df.columns if c not in drop_features]
    return df[feats], df['Churn']
x, y = preprocessing(df)
x.head(3)


# <h3>Split data</h3>
# 
# Divide our data in 80% for training and 20% for our final test.

# In[ ]:


# Split dataset in train and test
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=50)
# Create a training and testing dataset (lightgbm api object)
train_set = lgb.Dataset(data= train_x, label= train_y, silent=-1)
test_set = lgb.Dataset(data= test_x, label= test_y, silent=-1)


# <h2>2. Hyperparameter tunning</h2>
# 
# We have to define many hyperparameters since we have the ensemble estimator and the base estimator. Instead of doing this by hand, the next function will randomly pick a set of parameters and test it with a KFold cross-validation scheme. The LightGBM library has a built-in function to easily implement this validation method.
# 
# The procedure can be described with the following steps:
# * Randomly pick a subset of hyperparameters
# * Train and validate our model with KFold cross-validation (second function)
# * Repeat this procedure MAX_EVALS times
# * Get the best cross-validation score and hyperparameters
# 
# Early stopping will be used to stop the algorithm when the validation score doesn't improve for 50 boosting rounds.

# In[ ]:


def hyperparameter_random_search(train_set, params_grid, fixed_params, max_evals,
                                 num_folds, print_params=False):
    """Random search for hyperparameter optimization"""
    # Dataframe for results
    results = pd.DataFrame(columns = ['score', 'params', 'iteration'],
                                  index = list(range(max_evals)))
    
    # Keep searching until reach max evaluations
    for i in range(max_evals):
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in params_grid.items()}
        hyperparameters.update(fixed_params)
        if print_params:
            print(hyperparameters)
        # Evaluate randomly selected hyperparameters
        eval_results = objective(train_set, hyperparameters, i, num_folds)
        # Add results to our dataframe
        results.loc[i, :] = eval_results
    # Sort with best score on top
    results.sort_values('score', ascending=False, inplace=True)
    results.reset_index(inplace=True)
    return results

def objective(train_set, hyperparameters, iteration, num_folds):
    """Objective function for grid and random search. Returns
       the cross-validation score from a set of hyperparameters."""

     # Perform n_folds cross validation
    hyperparameters['verbose'] = -1
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round=10000, nfold=num_folds,
                        early_stopping_rounds=50, metrics ='auc', seed=50)
    score = cv_results['auc-mean'][-1]
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators 
    return score, hyperparameters, iteration


# <h2>3. Random Forests</h2>
# 
# The ideia behind Random Forests is to build multiple decision trees and merges them together to get a more accurate and stable prediction. In each estimator, only a random subset of the features is taken into consideration for splitting a node, therefore adding additional randomness to the final model. Random Forests can be considered an averaging ensemble method.

# In[ ]:


params_grid = {
    'num_leaves': list(range(8, 255)),  # Control tree size
    # Percentage (sample) of columns and rows
    'colsample_bytree': list(np.linspace(0.4, 0.99)),
    'subsample': list(np.linspace(0.4, 0.99)),
    # Min data points to create a leaf
    'min_child_samples': list(range(1, 101, 5)),
    # Regularization
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
}

fixed_params = {'boosting_type': 'rf', 'objective': 'binary',
                'subsample_freq': 1, 'n_jobs':4}
res = hyperparameter_random_search(train_set, params_grid, fixed_params, 1000, 5)
res.head()


# <h3>Testing the model</h3>

# In[ ]:


# Create, train, test model
model = lgb.LGBMClassifier(**res.loc[0, 'params'], random_state=50)
model.fit(train_x.values, train_y.values)
predict_proba = model.predict_proba(test_x.values)[:, 1]
predict_labels = model.predict(test_x.values)

# Print final results
print("Scores on test set: {:.4f} ROC AUC,  {:.4f} accuracy, {:.4f} recall, {:.4f} precision"
      .format(roc_auc_score(test_y, predict_proba),
              accuracy_score(test_y, predict_labels),
              recall_score(test_y, predict_labels), 
              precision_score(test_y, predict_labels)))


# <h2>4. Gradient Boosting</h2>
# 
# In gradient boosting, the predictors are not made independently, but sequentially. In each iteration, a decision tree is fit on the error from the previous round. It's usually better to use shallow trees (weak learners) and a low learning rate, so each iteration does small improvements to the overall problem. 

# In[ ]:


# Possible hyperparameters (grid)
param_grid = {
    'num_leaves': list(range(7, 95)),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2))),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.4, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
}
fixed_params = {'boosting': 'gbdt', 'objective': 'binary', 'n_jobs': 4}
res = hyperparameter_random_search(train_set, param_grid, fixed_params, 1000, 5)
res.head()


# <h3>Testing the model</h3>

# In[ ]:


# Create, train, test model
model = lgb.LGBMClassifier(**res.loc[0, 'params'], random_state=50)
model.fit(train_x.values, train_y.values)
predict_proba = model.predict_proba(test_x.values)[:, 1]
predict_labels = model.predict(test_x.values)

# Print final results
print("Scores on test set: {:.4f} ROC AUC,  {:.4f} accuracy, {:.4f} recall, {:.4f} precision"
      .format(roc_auc_score(test_y, predict_proba),
              accuracy_score(test_y, predict_labels),
              recall_score(test_y, predict_labels), 
              precision_score(test_y, predict_labels)))


# <h2>5. Conclusion</h2>
# 
# In this short notebook, we compared the performance of two ensemble methods in a binary classification problem. The first algorithm (Random Forests) is an averaging ensemble technique that train multiple decision trees in parallel and combine their predictions. Our second classifier was a boosting ensemble method, also using decision trees as base estimator. 
# 
# Hyperparameters were found using random search with a KFold validation scheme over more than one thousand tries for each estimator.

# In[ ]:




