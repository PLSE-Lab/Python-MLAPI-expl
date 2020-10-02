#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import lightgbm as lgbm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence

rand_seed = 13579
np.random.seed(rand_seed)

sns.set(style="darkgrid", context="notebook")

import os
print(os.listdir("../input"))
print(os.listdir("../input/train"))
print(os.listdir("../input/test"))


# In[ ]:


train_df = pd.read_csv("../input/train/train.csv")
test_df = pd.read_csv("../input/test/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


X_cols = ["Type", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize", 
          "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee", "State", 
          "VideoAmt", "PhotoAmt"]


# In[ ]:


def kappa_score(y1, y2, labels=None, sample_weight=None):
    return cohen_kappa_score(y1, y2, labels=labels, weights="quadratic", sample_weight=sample_weight)
kappa_scorer = make_scorer(kappa_score)


# In[ ]:


def lgbm_classifier_loss_function(params):
    num_leaves, max_depth, learning_rate, n_estimators, min_split_gain, min_child_weight, min_child_samples, subsample, subsample_freq, colsample_bytree, reg_alpha, reg_lambda = params
    
    lgbm_classifier = lgbm.LGBMClassifier(boosting_type="gbdt", num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, 
                                          n_estimators=n_estimators, subsample_for_bin=400000, objective="multiclass", class_weight=None, 
                                          min_split_gain=min_split_gain, min_child_weight=min_child_weight, min_child_samples=min_child_samples, 
                                          subsample=subsample, subsample_freq=subsample_freq, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, 
                                          reg_lambda=reg_lambda, random_state=rand_seed, n_jobs=-1, silent=True, importance_type="split")
    
    scores = cross_val_score(lgbm_classifier, train_df[X_cols], train_df["AdoptionSpeed"], cv=6, scoring=kappa_scorer, fit_params={"eval_metric":kappa_score}, error_score=-1.0)
    
    return -np.mean(scores) # Since skopt minimizes and 1.0 is the max, just flip the metric


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nsearch_space = [\n    Integer(10, 100), # num_leaves\n    Integer(5, 100), # max_depth\n    Real(1e-3, 1.0), # learning_rate\n    Integer(25, 250), # n_estimators\n    Real(0.0, 1.0), # min_split_gain\n    Real(1e-5, 1e-1), # min_child_weight\n    Integer(1, 40), # min_child_samples\n    Real(0.0, 1.0), # subsample\n    Integer(0, 50), # subsample_freq\n    Real(0.0, 1.0), # colsample_bytree\n    Real(0.0, 1e3), # reg_alpha\n    Real(0.0, 1e3), # reg_lambda\n]\n\nlgbm_res = gp_minimize(lgbm_classifier_loss_function, search_space, base_estimator=None, n_calls=750, n_random_starts=50, \n                  acq_func="gp_hedge", acq_optimizer="auto", x0=None, y0=None, random_state=rand_seed, verbose=True, \n                  callback=None, n_points=10000, n_restarts_optimizer=5, xi=0.01, kappa=1.96, noise="gaussian", n_jobs=1)')


# In[ ]:


num_leaves, max_depth, learning_rate, n_estimators, min_split_gain, min_child_weight, min_child_samples, subsample, subsample_freq, colsample_bytree, reg_alpha, reg_lambda = lgbm_res.x
lgbm_classifier = lgbm.LGBMClassifier(boosting_type="gbdt", num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, 
                                      n_estimators=n_estimators, subsample_for_bin=400000, objective="multiclass", class_weight=None, 
                                      min_split_gain=min_split_gain, min_child_weight=min_child_weight, min_child_samples=min_child_samples, 
                                      subsample=subsample, subsample_freq=subsample_freq, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, 
                                      reg_lambda=reg_lambda, random_state=rand_seed, n_jobs=-1, silent=True, importance_type="split")
lgbm_classifier.fit(train_df[X_cols], train_df["AdoptionSpeed"], eval_metric=kappa_score)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(6.0, 6.0)

plot_convergence(lgbm_res, ax=ax)

plt.show()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(12.0, 14.0)

lgbm.plot_importance(lgbm_classifier, ax=ax)

plt.show()


# In[ ]:


y_pred = lgbm_classifier.predict(test_df[X_cols])
submission_df = pd.DataFrame(data={"PetID":test_df["PetID"], "AdoptionSpeed":y_pred})
submission_df.to_csv("submission.csv", index=False)


# In[ ]:




