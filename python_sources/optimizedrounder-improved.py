#!/usr/bin/env python
# coding: utf-8

# This kernel is to demostrate an improved version of the OptimizedRounder function initially shared by Abhishek: https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/76107

# In[ ]:


# init
import os
print(os.listdir("../input"))

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score


# In[ ]:


# read data
df = pd.read_csv("../input/train/train.csv")

# using all variables for a basic model
Xvars = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength',
         'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'VideoAmt', 'PhotoAmt']

# basic rf model
import lightgbm as lgb
from functools import partial
import scipy as sp

params = {}
params['objective'] = 'regression'
params['boosting_type'] = 'rf'

params['min_data_in_leaf'] = 1
params['feature_fraction'] = 0.99
params['bagging_fraction'] = 0.4
params['bagging_freq'] = 50

params['num_threads'] = 4


# In[ ]:


# improved
class OptimizedRounder_v2(object):
    def __init__(self):
        self.coef_ = 0
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights = 'quadratic')
    
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')
    
    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return preds
    
    def coefficients(self):
        return self.coef_['x']


# # What are the improvements?
# 
# ### 1. The pd.cut() function does the job of mapping outputs to [0,1,2,3,4] in a very neat way, runs 2.5x faster
# > [-np.inf] + list(np.sort(coef)) + [np.inf]
# **      -->   This is because the outer bound conditions need to be specified**
# 
# ### 2. Solved a potential bug (explained below)
# * The original optimizer did not always preserve the order of the 4 coefficient values
# * Improved version always sorts coefficients before mapping prediction output to [0,1,2,3,4] (conveniently, pd.cut() would fail if the coefficients weren't in ascending order)
# * The final prediction kappa is not affected majorly, the optimizer still manages to converge well

# In[ ]:


# original version
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        
        print(coef)
        ll = cohen_kappa_score(y, X_p, weights = 'quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


# ## The bug in original version:
# Below is a print of each iteration of the optimizer.
# Where highlighted, you see that the coefficients are not in ascending order anymore.
# 
# `coef[0] < coef[1] < coef[2] < coef[3]` should be true in all cases
# 
# But in the highlighted lines, `coef[2] > coef[3]` which is not correct.
# 
# This then affects the step where values are mapped to [0,1,2,3,4] based on a for-loop if-else sequence.
# The kappa metric calculation gets thrown off course, and then confuses the optimizer in further iterations.
# 
# ![](https://i.imgur.com/FBEly8l.png)

# In[ ]:


# full optimizer output
def test():
    cv = KFold(n_splits = 5, random_state = 0)
    for train, val in cv.split(df):
        lgbDf = lgb.Dataset(df.iloc[train][Xvars], df.iloc[train].AdoptionSpeed)
        model = lgb.train(params, lgbDf, 1001)

        optR = OptimizedRounder()
        optR.fit(model.predict(df.iloc[train][Xvars]), df.iloc[train].AdoptionSpeed)
        coeff = optR.coefficients()
        break

start = time.time()
test()
finish = time.time()


# ## Speed comparison:

# In[ ]:


# original
print(finish - start)


# In[ ]:


# improved
def test_v2():
    cv = KFold(n_splits = 5, random_state = 0)
    for train, val in cv.split(df):
        lgbDf = lgb.Dataset(df.iloc[train][Xvars], df.iloc[train].AdoptionSpeed)
        model = lgb.train(params, lgbDf, 1001)

        optR = OptimizedRounder_v2()
        optR.fit(model.predict(df.iloc[train][Xvars]), df.iloc[train].AdoptionSpeed)
        coeff = optR.coefficients()
        break

start = time.time()
test_v2()
finish = time.time()
print(finish - start)


# ## What next?
# 
# How do we best initialize the coefficients? The optimizer seems very sensitive to how the coefficients are initialized.
# I don't have good insights on how nelder-mead works, but it seems to lock on to local optima.
# How could we best direct it to arrive at a more stable global optimum?
# (One great idea here from Daniel Dewey: https://www.kaggle.com/dan3dewey/baseline-random-forest-with-get-class-bounds)
# 
# In an ideal world, we should arrive at the same final coefficients irrespective of how they were initalized.
# If you've given this any thought, how are you approaching it?
# 
# Would love to hear your thoughts and feedback.
