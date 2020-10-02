#!/usr/bin/env python
# coding: utf-8

# # winPlacePct as ranked prediction
# I've noticed that in practise `winPlacePct` is:
# 
# - fixed within a group
# - within each game the group scores spam from 0 to 1 (inclusive) with a step of `1/numGroups`.
# 
# The conclusion is that in practice we need to predict **the order** of places for teams (=groups) **within each game(=match)**. It might have been obvious for everyone, but not to me. 
# 
# Major consequences are:
# 
# - it seems to be meaningful and beneficial **to train on group-level instead of the user-level**
# - it would be useful to be able to put constraints in training to fix the range to [0,1] within each game
# - as post-processing, it is useful to **calculate ranks** and use those as predictions

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import mean_absolute_error

import warnings
warnings.simplefilter(action='ignore', category=Warning)

from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
print(os.listdir("../input"))

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# As predictions, I will use output of my user-level kernel, that procudes OOF predictions together with the submission predictions: https://www.kaggle.com/mlisovyi/pubg-survivor-kit
# 
# This allows us to estimate the evaluation metric on OOF and thus to judge about expected level of improvement

# In[ ]:


df_trn = pd.read_csv('../input/pubg-finish-placement-prediction/train.csv',  nrows=None)
df_trn = reduce_mem_usage(df_trn)

df_tst = pd.read_csv('../input/pubg-finish-placement-prediction/test.csv',  nrows=None)
df_tst = reduce_mem_usage(df_tst)


# ### Read predictions for the user-level model (OOF and submission)

# In[ ]:


lgbm_trn = pd.read_csv('../input/pubg-survivor-kit/oof_lgbm1_reg.csv')
lgbm_trn.columns = [c if 'winPlacePerc' not in c else c+'Pred' for c in lgbm_trn.columns]

lgbm_tst = pd.read_csv('../input/pubg-survivor-kit/sub_lgbm1_reg.csv')
lgbm_tst.columns = [c if 'winPlacePerc' not in c else c+'Pred' for c in lgbm_tst.columns]


# Merge predictions with the main datasets

# In[ ]:


df_trn2 = pd.concat([df_trn, lgbm_trn['winPlacePercPred']], axis=1)
df_tst2 = pd.concat([df_tst, lgbm_tst['winPlacePercPred']], axis=1)


# # Rank mean predictions for each group within each game (train)

# Note that here we use an average over player ranking within each team (and that's what many people in public kernels copied over). This is a custom choice. 
# 
# It was checked that average performed better than median, min or max.

# In[ ]:


df_trn3 = df_trn2.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
df_trn3.columns = [c if 'winPlacePerc' not in c else c+'_Rank' for c in df_trn3.columns]
df_trn3 = df_trn3.merge(df_trn2, how='left', on=['matchId','groupId'])


# In[ ]:


print('MAE by default: {:.4f}'.format(
    mean_absolute_error(df_trn3['winPlacePerc'], df_trn3['winPlacePercPred'])
                                 )
     )


# In[ ]:


print('MAE after group ranking: {:.4f}'.format(
    mean_absolute_error(df_trn3['winPlacePerc'], df_trn3['winPlacePercPred_Rank'])
                                 )
     )


# **After group ranking the MAE metric significantly improves**

# # Rank mean predictions for each group within each game (test)

# In[ ]:


df_tst3 = df_tst2.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
df_tst3.columns = [c if 'winPlacePerc' not in c else c+'_Rank' for c in df_tst3.columns]
df_tst3 = df_tst2.merge(df_tst3, how='left', on=['matchId','groupId'])


# ## Store submission

# In[ ]:


del lgbm_tst['winPlacePercPred']
lgbm_tst['winPlacePerc'] = df_tst3['winPlacePercPred_Rank']


# In[ ]:


lgbm_tst.to_csv('sub_lgbm_group_ranked_within_game.csv', index=False)


# In[ ]:


get_ipython().system('head sub_lgbm_group_ranked_within_game.csv')


# In[ ]:




