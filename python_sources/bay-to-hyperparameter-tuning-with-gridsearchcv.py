#!/usr/bin/env python
# coding: utf-8

# # COMPREHENSIVE GUIDE TO HYPERPARAMETER TUNING
# [Vikum Wijesinghe](https://www.linkedin.com/in/vikumwijesinghe/) - September 2019
# 
# Other Kernels: https://www.kaggle.com/vikumsw/kernels
# 
# ---

# # Problem Description
# 
# ### There is an <font color="NAVY">ANGRY BABY</font>. What <font color="NAVY">FLAVOR</font> of <font color="NAVY">ICE CREAM</font> would you think is the best to make the baby pleased?
# 
# |||
# |:-:|:-:|
# |![](https://66.media.tumblr.com/bc5323888a02bb3c59e1d0f2b2e4ad32/3e0f261eb045a6d5-30/s250x400/cb65b307491d86d44614b7b8db2d8073832ce09e.gif)|![](https://www.executiveinnoakland.com/uploads/1/0/6/8/106825145/icecreamflavors-gvwrh9_orig.gif)
# 
# ### Analogy:
# * Angry Baby   -> Problem
# * Ice Cream    -> Choosen ML Algorithm to solve the problem
# * Flavors      -> Configurations/Properties of the ML Algorithm
# 
# 
# 
# ### How are you going to solve the problem?
# 1. Choose the most popular flavor :-> <font color="GRAY">Using your Machine Learning algorithm with default hyperparameters, You would most likely end up with a suboptimal model.</font>
# 2. Choose the flavor baby likes the most :-> <font color="GRAY">Tuning your models Hyperparameters to get the most skillful model.</font>
# 
# 
# Keep Reading to find out how to choose the flavor baby likes the most

# # Table Of Contents
# 
# 1. [What are hyperparameters?]()
# 1. [Some examples of hyperparameters]()
# 1. [Why they are important?]()
# 1. [How to tune hyperparamters?]()
# 1. [generate a regression problem for the exercise]()
# 1. [Hyperparameter Tuning Using Grid Search]()

# #### Libray Imports

# In[ ]:


from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import os
import sys
from IPython.display import Image


# ## What are hyperparameters?
# 
# Let's create a Random Forest Regressor model for demonstration,

# These parameters express configurations of the model such as its structure or learning rates. They are called hyperparameters.
# These values cannot be estimated from data. So hyperparameters are usually set before training. Think of it like there exist different flavors of the same machine learning algorithm.
# 
# ![](https://media.giphy.com/media/QmKtZGQn4cNi6EC15Y/giphy.gif)
# 
# ## Some examples of hyperparameters :
# * Number of leaves or depth of a tree
# * Learning rate
# * Number of hidden layers in a deep neural network
# * Number of clusters in a k-means clustering
# 
# ## Why are they important?.
# 
# In addition to choosing the best suited Machine Learning model for a particular problem, selecting the best flavour of the selected model also decides the performance.
# 
# ## How to tune hyperparamters?
# There are several ways of choosing a set of optimal hyperparameters for a learning algorithm.
# * Grid search
# * Manual search 
# * Random search
# * Bayesian Optimization and More..
# 
# I will stick to grid search in this discussion, links to study others are provided in the bottom of the notebook. 

# ## Data for the exercise
# 
# For this exercise let's generate a random regression problem using sklearn.datasets.make_regression.
# 
# [Want to know more on make_regression?. have a look..](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)

# In[ ]:


from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=4, n_informative=2,random_state=0, shuffle=False)

f,ax=plt.subplots(2,2,figsize=(14,14))

sns.scatterplot(x=X[:,0], y=y, ax=ax[0,0])
ax[0,0].set_xlabel('Feature 1 Values')
ax[0,0].set_ylabel('Y Values')
ax[0,0].set_title('Sactter Plot : Feature 1 vs Y')

sns.scatterplot(x=X[:,1], y=y,ax=ax[0,1])
ax[0,1].set_xlabel('Feature 2 Values')
ax[0,1].set_ylabel('Y Values')
ax[0,1].set_title('Sactter Plot : Feature 2 vs Y')

sns.scatterplot(x=X[:,2], y=y,ax=ax[1,0])
ax[1,0].set_xlabel('Feature 3 Values')
ax[1,0].set_ylabel('Y Values')
ax[1,0].set_title('Sactter Plot : Feature 3 vs Y')

sns.scatterplot(x=X[:,3], y=y,ax=ax[1,1])
ax[1,1].set_xlabel('Feature 4 Values')
ax[1,1].set_ylabel('Y Values')
ax[1,1].set_title('Sactter Plot : Feature 4 vs Y')

plt.show()


# ## Hyperparameter tuning using grid search
# 
# > The traditional way of performing hyperparameter optimization has been grid search, or a parameter sweep, which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm. A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set or evaluation on a held-out validation set.
# ([Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search))

# Now Data is ready for training... First we need a regressor ... lets choose RandomForestRegressor... and see default values of hyperparamters

# In[ ]:


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

rfr = RandomForestRegressor(verbose=0)
print('Parameters currently in use:\n')
pprint(rfr.get_params())
print('CV score with default parameters : ',-cross_val_score(rfr, X, y, cv=4, scoring = make_scorer(mean_squared_error, greater_is_better=False)).mean())


# Thats default parameters. Our target is to find the best set of hyperparameters from a subset taken from hyperparameter space.
# For this we need to,
# * Define the parameter grid : -> Subset from the hyperparameter space to search.
# * Make a scorer : -> This scorer will be use to choose the best performing model.
# 
# Then we create the GridSearch model using sklearn.model_selection.GridSearchCV and proceed to trainig. Let's see how its done.

# In[ ]:


from sklearn.model_selection import GridSearchCV

# define the search space.
param_grid = {
    'bootstrap': [True],
    'max_depth': [50, 75, 100],
    'max_features': ['auto'],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'n_estimators': [100,200,500,1000]}

# make scorer
MSE = make_scorer(mean_squared_error, greater_is_better=False)

# Configure the GridSearch model
model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=-1, cv=4, scoring=MSE, verbose=2)
# n_jobs=-1 : Means configured to use parallelism. use n_jobs=1 if use wish not to.

# Training
model.fit(X, y)

print('Random forest regression...')
print('Best Params:', model.best_params_)
print('Best CV Score:', -model.best_score_)


# Explanation : 
# We have scored random forest regressor for following hyperparameter settings.
# 
# | bootstrap | max_depth  | max_features  | min_samples_leaf  |  min_samples_split | n_estimators  |
# |:-:|:-:|:-:|:-:|:-:|---|
# | True  | 50 |  'auto' |  1 | 2  |  100 |
# | True  | 50 |  'auto' |  1 | 2  |  100 |
# | True  | 50 |  'auto' |  1 | 2  |  100 |
# | True  | 50 |  'auto' |  1 | 2  | 200 |
# | True  | 75 |  'auto' |  1 | 2  |  200 |
# | True  | 75 |  'auto' |  1 | 2  |  200 |
# | True  | 75 |  'auto' |  1 | 2  |  5000 |
# | True  | 75 |  'auto' |  1 | 2  |  5000 |
# | True  | 100 |  'auto' |  1 | 2  |  5000 |
# | True  | 100 |  'auto' |  1 | 2  |  1000 |
# | True  | 100 |  'auto' |  1 | 2  |  1000 |
# | True  | 100 |  'auto' |  1 | 2  |  1000 |
# 
# ### Next??
# Now since we have found the best random forest regressor inside our hyperparameter search space for this particular problem. we can proceed to prediction.
# We can simple use,
# 
#     Y_predictions = model.predict(X)

# ### Solving the problem using XGBRegressor

# In[ ]:


import xgboost as xgb

xgbr = xgb.XGBRegressor(seed=0)
# A parameter grid for XGBoost
param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


model = GridSearchCV(estimator=xgbr, param_grid=param_grid, n_jobs=-1, cv=4, scoring=MSE)
model.fit(X, y)
score = -model.best_score_

print('eXtreme Gradient Boosting regression...')
print(xgbr)
print('Best Params:\n', model.best_params_)
print('Best CV Score:', score)


# Looks like XGBRegressor is way better than the random Forest Regressor.. And that's a another kind of pleaser.. Let's say Choclate.  Although we found out that choclate is betterto calm the baby than ice cream, we only got ice cream. so lets proceed with the best ice cream flavor...

# ### Let's see how it goes!. Time to calm the Baby...
# 
# ![](https://media.giphy.com/media/AGGz7y0rCYxdS/giphy.gif)
# 
# ### Oh How Cute! Bravo!!!. Given the best solution!

# ## Learn More
# * [What are hyperparameters in machine learning?](https://www.quora.com/What-are-hyperparameters-in-machine-learning)
# * [What is the Difference Between a Parameter and a Hyperparameter?](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)
# * [Hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)

# ## Please upvote if you found it useful and joyful!
