#!/usr/bin/env python
# coding: utf-8

# # Inequality regarding WRMSSE
# In this notebook, I would like to show the inequality of WRMSSE.
# 
# 
# 
# 

# ## Interpretion on WRMSSE
# Weighted Root Mean Squared Scaled Error (WRMSSE) is defined as
# 
# \begin{equation}
# \mathrm{WRMSSE} = \sum_{i} w_i \sqrt{ \sum_{t=n+1}^{n+h} \frac{1}{h} \frac{(y_{it} - \hat{y}_{it})^2}{S_i}},
# \end{equation}
# 
# where $S_i$ means random walk, 
# 
# \begin{equation}
# S_i = \frac{1}{n-1} \sum^n_{t=2} (y_{it} - \hat{y}_{it})^2.
# \end{equation}
# 
# With some calculation, 
# 
# \begin{equation}
# \mathrm{WRMSSE} = \sum_{i} \frac{w_i}{\sqrt{S_i}} \sqrt{ \sum_{t=n+1}^{n+h} \frac{1}{h} (y_{it} - \hat{y}_{it})^2}.
# \end{equation}
# 
# So, we can regard $ w_i / \sqrt{S_i}$ as the net weight.
# This factor can be interpreted as price times randomness.

# ## Inequality regarging WRMSSE
# Using triangle inequality $\sqrt{\sum_i a_i^2} \leq \sum_i |a_i|$
# \begin{equation}
# \sqrt{ \sum_t (y_{it} - \hat{y}_{it})^2 } \leq \sum_t |y_{it} - \hat{y}_{it}|.
# \end{equation}
# 
# Thus, 
# \begin{equation}
# \mathrm{WRMSSE} \leq \sum_{i, t} \frac{w_i}{\sqrt{h S_i}} |y_{it} - \hat{y}_{it}|.
# \end{equation}
# 
# Defined as 
# \begin{equation}
# z_{it} := w_i y_{it} / \sqrt{h S_i}, \\
# N := \sum_i 1, 
# \end{equation}
# 
# WRMSSE can be evaluated as
# \begin{equation}
# \mathrm{WRMSSE} \leq hN \times \frac{1}{hN} \sum_{i,t} |z_{it} - \hat{z}_{it} |.
# \end{equation}
# 
# The right side of this inequality can be interpreted as Mean Absolute Error (MAE).
# Similar procedures are found in [this paper](https://robjhyndman.com/papers/mase.pdf), so I would like to call this one psuedo-Weighted Mean Absolute Scaled Error (psuedo-WMASE).
# 
# Since WRMSSE is bounded above by psuedo-WMASE, WRMSSE is always lower than psuedo-WMASE.
# 
# In this expression, we don't have to care weights and scaling factors when predictions are evaluated 
# because sum about indices i and t can be exchanged.
# 
# It is difficult to adopt WRMSSE directly, but after these transformations, we can use psuedo-WMASE as metric easily. 
# When you take some transformation, true validation score for WRMSSE in training is bounded above by psuedo-WMASE.
# 
# (TRUE Weighted Mean Absolute Scaled Error (WMASE) may be defined as
# \begin{equation}
# \mathrm{WMASE} = \sum_{i} w_i \sum_{t=n+1}^{n+h} \frac{1}{h} \frac{|y_{it} - \hat{y}_{it}|}{S'_i},
# \end{equation}
# where
# \begin{equation}
# S'_i = \frac{1}{n-1} \sum^n_{t=2} |y_{it} - \hat{y}_{it}|.
# \end{equation}
# )

# ## Difficulty in this competition
# In this competition, we must predict not only one aggregated level but other aggregated or disaggregated levels.
# $y_{it}$ depends on other levels. 
# 
# If we take bottom-up approach, all of upper level values are determined automatically.
# So are if you take top-down or middle-out approach.
# (Of course, this is too naive. 
# There would be many approaches that overcomes this difficulty coming from this level-dependent metric).
# 
# However, at least, we can know if the predictions at some level are good or not while training with validation sets.

# ## How to use this idea (example)
# Thanks to the help of some great notebooks ([M5 - Simple FE](https://www.kaggle.com/kyakovlev/m5-simple-fe) [@kyakovlev](https://www.kaggle.com/kyakovlev) and [Fast & Clear WRMSSE 18ms](https://www.kaggle.com/sibmike/fast-clear-wrmsse-18ms) [@sibmike](https://www.kaggle.com/sibmike)), I would show the idea above with the simple bottom-up approach.
# 
# * train: d_730 - d_1885
# * validation: d_1886 - d_1913
# * test: d_1914 - d_1941
# 
# The steps to apply the idea above are:
# 1. Make $z_{it}$ with 'sales', weights, scaling factors.
# 2. Train and predict (Target = $z_{it}$).
# 3. Make 'sales' prediction at the bottom level from $z_{it}$

# In[ ]:


import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import warnings

warnings.filterwarnings('ignore')


# In[ ]:


## file path ##
BASE = '../input/m5-simple-fe/grid_part_1.pkl'
CALENDAR = '../input/m5-simple-fe/grid_part_3.pkl'
LAGS = '../input/m5-lags-features/lags_df_28.pkl'

SW = '../input/fast-clear-wrmsse-18ms/sw_df.pkl'


# In[ ]:


## basic features ##
df = pd.concat([pd.read_pickle(BASE),
                    pd.read_pickle(CALENDAR).iloc[:,9:]],
                    axis=1)

#df = pd.read_pickle(BASE)

## lag features ##
lag_df = pd.read_pickle(LAGS)
lag_df = lag_df.iloc[:, 3:11]

## input data ##
grid_df = pd.concat([df, lag_df], axis=1)

del lag_df, df
gc.collect()


# In[ ]:


## weights and scaling factors ##
# s, w, sw in sw_df are scaling factor, weight, and the product of them respectively.
# Since we use only the product sw, other columns are dropped.

sw_df = pd.read_pickle(SW)

sw_df.reset_index(inplace=True)
sw_df = sw_df[sw_df.level==11]
sw_df.drop(['level', 's', 'w'], axis=1, inplace=True)

sw_df['id'] = sw_df['id'].astype('category')
grid_df = grid_df.merge(sw_df, on='id', how='left')

# The product of sales and sw corresponds to z_it (different by a factor).
# This one is the main target.
grid_df['sw_sales'] = grid_df['sales'] * grid_df['sw']

del sw_df
gc.collect()


# In[ ]:


## training model (LGBM) ##

# train, validation and test set
START_TRAIN = 730
END_TRAIN = 1913
P_HORIZON = 28

grid_df = grid_df[grid_df.d>=START_TRAIN]
grid_df.to_pickle('grid_df_ex.pkl')

test_idx = grid_df.d > END_TRAIN
valid_idx = (grid_df.d <= END_TRAIN) & (grid_df.d > END_TRAIN - P_HORIZON)
train_idx = (grid_df.d <= END_TRAIN- P_HORIZON) & (grid_df.d >= START_TRAIN)


# hyper parameters
lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'learning_rate': 0.1,
                    'num_leaves': 2**5-1,
                    'min_data_in_leaf': 2**6-1,
                    'n_estimators': 100,
                    'boost_from_average': False,
                    'verbose': -1,
                } 

# Set the metric in training Mean Absolute Error (MAE).
# Using MAE with target 'sw_sales', validation values in training show psurdo-WMASE.
lgb_params['metric'] = 'mae'


# In[ ]:


get_ipython().system('rm train_data.bin')


# In[ ]:


## Indirect Prediction ##

# features and target
remove_fe = ['id', 'd', 'sales', 'sw', 'sw_sales']
features = [fe for fe in list(grid_df) if fe not in remove_fe]

TARGET = 'sw_sales'

# dataset
train_data = lgb.Dataset(grid_df[train_idx][features], 
                        label=grid_df[train_idx][TARGET])
train_data.save_binary('train_data.bin')
train_data = lgb.Dataset('train_data.bin')

valid_data = lgb.Dataset(grid_df[valid_idx][features],
                        label=grid_df[valid_idx][TARGET])

del grid_df
gc.collect()

# model training
estimator = lgb.train(lgb_params,
                      train_data,
                      valid_sets = [train_data, valid_data],
                      verbose_eval = 10,
                      early_stopping_rounds = 5,
                      )

# Validaiton result means psuedo-WMASE at the bottom level (level 12).
# Calculated psuedo-WMASE is different by a constant factor.
# The validation score means 


# The validation score in training is $\frac{1}{hN} \sum_i |z_{it} - \hat{z}_{it}|$. To get psuedo-WMASE, we must the factor $hN$, but this is just a constant value ($h=28, N=30490$). 
# So I ignore the difference.

# In[ ]:


## prediction for test set ##
grid_df = pd.read_pickle('grid_df_ex.pkl')

test_data = grid_df[test_idx][features]
grid_df['sw_sales'][test_idx] = estimator.predict(test_data)

del test_data
gc.collect()

# sw_sales -> sales
grid_df['sales'][test_idx] = grid_df['sw_sales'][test_idx] / grid_df['sw'][test_idx]


# In[ ]:


# The final output 'sales' accuracy is bound above at the bottom level.
# However, there are some items with weight=0.
# Some item has sales values of infinity......
grid_df['sales'][test_idx].max()


# At the bottom level we don't have to care this infinity, but for upper aggregated level, we must predict these zero-weight items sales if we take bottom-up approach.
# 
# In that case, we need another model which predicts or complements these zero-weight items sales.
