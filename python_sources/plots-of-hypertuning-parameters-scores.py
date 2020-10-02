#!/usr/bin/env python
# coding: utf-8

# Tuning hyperparameters can be complex. Some of the parameters should be in the 0-to-1 range, others have to be more than 0, others - more than 1 and so on...
# 
# Also, even if you know these acceptable ranges, you should also remember how do the trends of the score look like with tuning each of the parameters.

# In[ ]:


#################################################################
#### Libs and globals ####

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore')

seed = 11

# libraries for cv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

# libraries for random search
from sklearn.model_selection import RandomizedSearchCV
from pandas.io.json import json_normalize
from matplotlib import pyplot as plt

# libs for timing
import time
import datetime

# iterations
n_iter=35
cv=5

#################################################################
#### Reading data and splitting it to target and features ####

train = pd.read_csv('../input/learn-together/train.csv', index_col="Id")
del train['Soil_Type15']
del train['Soil_Type7']

features = train.drop(['Cover_Type'], axis=1)
target = train.Cover_Type

def fit_and_draw(key, value):
    """ performing random search and returning a result set for different parameter values (with cv)"""
    param_dist = {key: value}
    
    start_time = time.time()

    random_search = RandomizedSearchCV(estimator=model,
                                 param_distributions=param_dist,
                                 scoring ='accuracy',
                                 n_iter=n_iter,
                                 cv=cv,
                                 n_jobs=-1, 
                                 verbose=0)
    
    random_search.fit(features, target)
    
    trial_time = time.time() - start_time 
    trial_time_per_search = trial_time/(cv * n_iter)
    trial_time = trial_time/60
    
    print('')
    print('')
    print(f'>>>> Results for {key}:')    
    print(f'> {(trial_time):.1f} minutes for search ({n_iter} iterations with {cv}-fold CV)')
    print(f'> {(trial_time_per_search):.1f} seconds per 1 search without CV')

    results = pd.DataFrame(random_search.cv_results_)[['params', 'mean_test_score']]

    params = pd.io.json.json_normalize(results['params'])

    result = (
        pd.merge(results, params, how='inner', left_index=True, right_index=True)
        .drop(columns=['params'])
        .groupby(key).mean()
    )

    return result

def demo():
    for key in param_dist:

        random_search = RandomizedSearchCV(estimator=model,
                                         param_distributions=param_dist,
                                         scoring ='accuracy',
                                         n_iter=n_iter,
                                         cv=cv,
                                         n_jobs=-1, 
                                         verbose=0)

        random_search.fit(features, target)

        # processing results
        results = pd.DataFrame(random_search.cv_results_)[['params', 'mean_test_score']]
        params = pd.io.json.json_normalize(results['params'])

        result = (
                pd.merge(results, params, how='inner', left_index=True, right_index=True)
                .drop(columns=['params'])
                .sort_values(by=key)
        )

        result.index = result[key]


        # printing plot
        ax = plt.subplot()
        plt.plot(result.index, result.mean_test_score)
        plt.title(key)
        plt.show()


# # Types of hyperparameters score trends
# 
# For expample, RandomForest's `n_estimators` have a positive effect on the score, but just until it reaches its asymptotic level.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=seed)
param_dist = {'n_estimators':np.arange(1, 250)}
demo()


# Other parameters like `n_neighbors` of KNeighborsClassifier peak the score on lower levels, and have a negative correlation with the score

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_jobs=-1)
param_dist = {'n_neighbors': np.arange(1, 50)}
demo()


# Some parameters like KNeighborsClassifier `leaf_size` can have no effect with defaults and can be valuable while using them with the appropriate set of other tuned params

# In[ ]:


param_dist = {'leaf_size' : np.arange(1, 150)}
demo()


# Some others have no strong trend, like, for instance, `intercept_scaling` in LogisticRegression

# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=seed, n_jobs=-1)
param_dist = {'intercept_scaling' : np.arange(0, 500)}
demo()


# That is why tuning parameters is going to be much easier if you know the ranges and usual trend of the score for the params.

# # Searching models
# 
# I've chosen several models to show the graphs of the score variation in relation tith diferent hyperparameters. Models are:
# * LogisticRegression
# * RidgeClassifier
# * RandomForestClassifier
# * KNeighborsClassifier
# * LGBMClassifier
# * XGBClassifier

# In[ ]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(random_state=seed, n_jobs=-1)

LR_params = {
    'tol': [1/(6**c) for c in range(1, 10)],
    'C' : np.arange(1, 250),
    'intercept_scaling' : np.arange(0, 500),
    'max_iter' : np.arange(0, 120) #(default=100)
}


from sklearn.linear_model import RidgeClassifier

Ridge = RidgeClassifier(random_state=seed)

Ridge_params = {
    'alpha': np.arange(0, 150)/100,
    'max_iter' : np.arange(0, 50),
    'tol': [1/(2**c) for c in range(1, 15)],
    'solver' : ['svd', 'cholesky','lsqr','sparse_cg']
}


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state=seed, n_estimators=50, n_jobs=-1)

RF_params = {
    'n_estimators': np.arange(5, 250),
    'max_depth' : np.arange(1, 50),
    'min_samples_split' : np.arange(1, 99)/100,
    'min_samples_leaf' : np.arange(1, 50)/100, 
    'min_weight_fraction_leaf' : np.arange(0, 50)/100,  
    'max_features' : np.arange(1, 100)/100, 
    'max_leaf_nodes' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_impurity_decrease' : np.arange(1, 99)/100,
}


from sklearn.neighbors import KNeighborsClassifier

KN = KNeighborsClassifier(n_jobs=-1)

KN_params = {
    'n_neighbors': np.arange(1, 50),
    'leaf_size' : np.arange(1, 150)    
}


from lightgbm import LGBMClassifier

LGBM = LGBMClassifier(random_state=seed, n_jobs=-1)

LGBM_params = {
    'num_leaves ': np.arange(1, 100),
    'max_depth' : [2*c for c in range(0, 25)],
    'learning_rate' : [1/(1.8**c) for c in range(1, 15)], 
    'n_estimators' : np.arange(1, 150),  
    'min_split_gain' : [1/(1.4**c) for c in range(1, 15)],
    'min_child_weight' : [1/(2**c) for c in range(1, 15)],
    'min_child_samples' : np.arange(1, 150),       
    'subsample' : [1/(2**c) for c in range(0, 5)],  
    'subsample_freq' : [(2**c) for c in range(0, 5)],   
    'colsample_bytree' : np.arange(1, 100)/100,
    'reg_alpha' : np.arange(0, 150)/100,
    'reg_lambda' : np.arange(0, 35)
}


from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=seed, verbosity=0, n_jobs=-1)

xgb_params = {
    'max_depth' : np.arange(0, 50),
    'learning_rate' : [1/(1.8**c) for c in range(1, 15)],  
    'n_estimators' : np.arange(1, 100),
    'gamma' : np.arange(0, 100),
    'min_child_weight' : np.arange(0, 50),
    'max_delta_step': np.arange(0, 100),
    'subsample': [1/(1.1**c) for c in range(1, 15)],
    'colsample_bytree' : [1/(1.1**c) for c in range(1, 15)],
    'colsample_bylevel' : [1/(1.1**c) for c in range(1, 15)],
    'colsample_bynode' : [1/(1.1**c) for c in range(1, 15)],
    'reg_alpha' : np.arange(0, 50),
    'reg_lambda' : np.arange(0, 50),
    'scale_pos_weight' : np.arange(0, 400)/100
}


################################################################

# drawing plots

models_and_params = {
    LR    :  LR_params,
    Ridge :  Ridge_params,    
    RF    :  RF_params,
    KN    :  KN_params,
#    LGBM : xgb_params,
#    xgb   :  xgb_params
}

session_start_time = time.time()

for key in models_and_params:

    model = key
    param_dist = models_and_params[key]

    print('*'*80)
    print('PRINTING PLOTS FOR MODEL:')
    print(key)
    
    for key in param_dist:
        result = fit_and_draw(key, param_dist[key])

        ax = plt.subplot()
        plt.plot(result.index, result.mean_test_score)
        plt.title(key)
        plt.show()

session_time = (time.time() - session_start_time) / 60

print('')
print('')
print('*'*80)
print(f'>>Drawing plots finished in {(session_time):.1f} minutes')


# I've chosen these models as they seemed to be one of the most popular linear, trees and boosting models. Similar models from the same algorithm group have similar parameters, so there was no need to do all of them because they have pretty same idea in tuning.
# 
# **Important - gboost models are commented**. Kernels just do not want to fit them, I don't know why. Running script on the local machine works perfectly well. It you'd like to try gboost models - just download the script and run it at home.
# 
# I will appreciate if someone will help me with learning gboost models on kaggle kernels. I can't figure out what am I doing wrong. 
# 
# I hope this notebook will help you with choosing the ranges of parameter tuning for your models.
# 
# Feel free to upvote!
