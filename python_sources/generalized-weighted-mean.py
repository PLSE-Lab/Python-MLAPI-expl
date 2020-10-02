#!/usr/bin/env python
# coding: utf-8

# 
# ## Generalized mean
# We use the [weighted generalized mean](https://en.wikipedia.org/wiki/Generalized_mean) to blend our predictions.  
# 
# $$
# \bar{x}
# = \left( \sum_{i=1}^n w_i x_i^p \right)^{1/p}
# $$
# 
# We tuned the weights and $p$ using the optuna library.

# In[ ]:


import os
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
from functools import partial
from sklearn.metrics import mean_squared_error


# In[ ]:


get_ipython().run_cell_magic('time', '', "root = '../input/ashrae-feather-format-for-fast-loading'\ntest = pd.read_feather(f'{root}/test.feather')\nmeta = pd.read_feather(f'{root}/building_metadata.feather')")


# In[ ]:


leak = pd.read_feather('../input/ashrae-leak-data-station/leak.feather')

leak.fillna(0, inplace=True)
leak = leak[(leak.timestamp.dt.year > 2016) & (leak.timestamp.dt.year < 2019)]
leak.loc[leak.meter_reading < 0, 'meter_reading'] = 0 # remove negative values
leak = leak[leak.building_id != 245]


# # Leak Validation for public kernels(not used leak data)

# In[ ]:


submission_list = [   
    "20191214-catboost-no-split-1-100-v2/2019-12-14_catboost-no_split_1-100_v2",
    "aggregate-models-v5/weighted_blend_2019-12-14_gbm_split_primary_use",
    "aggregate-models-v5/weighted_blend_2019-12-10_gbm_with_trend",
]

for i,f in enumerate(submission_list):
    x = pd.read_csv(f'../input/{f}.csv', index_col=0).meter_reading
    x[x < 0] = 0
    test[f'pred{i}'] = x

del  x


# In[ ]:


leak = pd.merge(leak, test[['building_id', 'meter', 'timestamp', *[f"pred{i}" for i in range(len(submission_list))], 'row_id']], "left")
leak = pd.merge(leak, meta[['building_id', 'site_id']], 'left')


# In[ ]:


for i in range(len(submission_list)):
    sns.distplot(np.log1p(leak[f"pred{i}"]))
    sns.distplot(np.log1p(leak.meter_reading))
    leak_score = np.sqrt(mean_squared_error(np.log1p(leak[f"pred{i}"]), np.log1p(leak.meter_reading)))
    print(f'score{i}={leak_score}')    


# # Leak Validation for Blending

# In[ ]:


# log1p then mean
log1p_then_mean = np.mean(np.log1p(leak[[f"pred{i}" for i in range(len(submission_list))]].values), axis=1)
leak_score = np.sqrt(mean_squared_error(log1p_then_mean, np.log1p(leak.meter_reading)))
print('log1p then mean score =', leak_score)


# In[ ]:


# mean then log1p
mean_then_log1p = np.log1p(np.mean(leak[[f"pred{i}" for i in range(len(submission_list))]].values, axis=1))
leak_score = np.sqrt(mean_squared_error(mean_then_log1p, np.log1p(leak.meter_reading)))
print('mean then log1p score=', leak_score)


# ## Tune with Optuna

# In[ ]:


class GeneralizedMeanBlender():
    """Combines multiple predictions using generalized mean"""
    def __init__(self, p_range=(-2,2)):
        """"""
        self.p_range = p_range
        self.p = None
        self.weights = None
                
    def _objective(self, trial, X, y):
                    
        # create hyperparameters
        p = trial.suggest_uniform(f"p", *self.p_range)
        weights = [
            trial.suggest_uniform(f"w{i}", 0, 1)
            for i in range(X.shape[1])
        ]

        # blend predictions
        blend_preds, total_weight = 0, 0
        if p <= 0:
            for j,w in enumerate(weights):
                blend_preds += w*np.log1p(X[:,j])
                total_weight += w
            blend_preds = np.expm1(blend_preds/total_weight)
        else:
            for j,w in enumerate(weights):
                blend_preds += w*X[:,j]**p
                total_weight += w
            blend_preds = (blend_preds/total_weight)**(1/p)
            
        # calculate mean squared error
        return np.sqrt(mean_squared_error(y, blend_preds))

    def fit(self, X, y, n_trials=10): 
        # optimize objective
        obj = partial(self._objective, X=X, y=y)
        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials)
        # extract best weights
        if self.p is None:
            self.p = [v for k,v in study.best_params.items() if "p" in k][0]
        self.weights = np.array([v for k,v in study.best_params.items() if "w" in k])
        self.weights /= self.weights.sum()

    def transform(self, X): 
        assert self.weights is not None and self.p is not None,        "Must call fit method before transform"
        if self.p == 0:
            return np.expm1(np.dot(np.log1p(X), self.weights))
        else:
            return np.dot(X**self.p, self.weights)**(1/self.p)
    
    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)


# In[ ]:


X = np.log1p(leak[[f"pred{i}" for i in range(len(submission_list))]].values)
y = np.log1p(leak["meter_reading"].values)

gmb = GeneralizedMeanBlender()
gmb.fit(X, y, n_trials=20)


# In[ ]:


print(np.sqrt(mean_squared_error(gmb.transform(X), np.log1p(leak.meter_reading))))


# # Submit

# In[ ]:


# make test predictions
sample_submission = pd.read_csv("/kaggle/input/ashrae-energy-prediction/sample_submission.csv")
X_test = test[[f"pred{i}" for i in range(len(submission_list))]].values
sample_submission['meter_reading'] = np.expm1(gmb.transform(np.log1p(X_test)))
sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0

# fill in leak data
leak = leak[['meter_reading', 'row_id']].set_index('row_id').dropna()
sample_submission.loc[leak.index, 'meter_reading'] = leak['meter_reading']

# save submission
sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')
sample_submission.head()


# In[ ]:


sns.distplot(np.log1p(sample_submission.meter_reading))

