#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, gc, warnings, random, math

from sklearn import metrics
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# In[ ]:


########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[ ]:


########################### Vars
#################################################################################
SEED = 42
seed_everything(SEED)
TARGET = 'meter_reading'


# In[ ]:


########################### DATA LOAD
#################################################################################
print('Load Data')
train_df = pd.read_pickle('../input/ashrae-data-minification/train.pkl')
building_df = pd.read_pickle('../input/ashrae-data-minification/building_metadata.pkl')
train_weather_df = pd.read_pickle('../input/ashrae-data-minification/weather_train.pkl')

test_df = pd.read_pickle('../input/ashrae-data-minification/test.pkl')


# In[ ]:


########################### Check building_id 
#################################################################################
temp_df = test_df[~test_df['building_id'].isin(train_df['building_id'])]
print('No intersection:', len(temp_df))
del test_df


# In[ ]:


########################### Merge additional data
#################################################################################
temp_df = train_df[['building_id']]
temp_df = temp_df.merge(building_df, on=['building_id'], how='left')
del temp_df['building_id']
train_df = pd.concat([train_df, temp_df], axis=1)

del building_df, temp_df

temp_df = train_df[['site_id','timestamp']]
temp_df = temp_df.merge(train_weather_df, on=['site_id','timestamp'], how='left')
del temp_df['site_id'], temp_df['timestamp']
train_df = pd.concat([train_df, temp_df], axis=1)

del train_weather_df, temp_df


# In[ ]:


########################### Model params
import lightgbm as lgb
lgb_params = {
                    'objective':'regression',
                    'boosting_type':'gbdt',
                    'metric':'rmse',
                    'n_jobs':-1,
                    'learning_rate':0.3, #for faster training
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'n_estimators':1000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                } 


# ----

# ## CV concept
# 
# ### Basics
# 
# > Cross-validation is a technique for evaluating ML models 
# > by training several ML models on subsets of the available 
# > input data and evaluating them on the complementary 
# > subset of the data. 
# 
# > In k-fold cross-validation, you split the input data 
# > into k subsets of data (also known as folds).
# 
# 
# ### Main strategy
# 1. Divide Train set in subsets (Training set itself + Validation set)
# 2. Define Validation Metric (in our case it is RMSE/RMSLE)
# 3. Stop training when Validation metric stops improving
# 4. Make predictions for Test set
# 
# Seems simple but he devil's always in the details.

# In[ ]:


########################### Create Holdout sets
#################################################################################
# Holdout set 1
# Split train set by building_id -> 20% to houldout
train_buildings, test_buildings = train_test_split(train_df['site_id'].unique(), test_size=0.20, random_state=SEED)

holdout_subset_1 = train_df[train_df['site_id'].isin(test_buildings)].reset_index(drop=True)
train_df = train_df[train_df['site_id'].isin(train_buildings)].reset_index(drop=True)

# Holdout set 2
# Split train set by site_id -> 20% to houldout                   
train_buildings, test_buildings = train_test_split(train_df['building_id'].unique(), test_size=0.20, random_state=SEED)

holdout_subset_2 = train_df[train_df['building_id'].isin(test_buildings)].reset_index(drop=True)
train_df = train_df[train_df['building_id'].isin(train_buildings)].reset_index(drop=True)
                    
# Holdout set 3
# Split train set by month -> first and last months to holdout
holdout_subset_3 = train_df[(train_df['DT_M']==1)|(train_df['DT_M']==12)].reset_index(drop=True)
train_df = train_df[(train_df['DT_M']!=1)&(train_df['DT_M']!=12)].reset_index(drop=True)

# Transform target and check shape
for df in [train_df, holdout_subset_1, holdout_subset_2, holdout_subset_3]:
    df[TARGET] = np.log1p(df[TARGET])
    print(df.shape)


# In[ ]:


########################### Features to use and eval sets
# for validation "purity" we will also remove site_id, building_id, DT_M
remove_columns = ['timestamp','site_id','building_id','DT_M',TARGET]
features_columns = [col for col in list(train_df) if col not in remove_columns]

X = train_df[features_columns]
y = train_df[TARGET]

split_by_building = train_df['building_id']
split_by_site = train_df['site_id']
split_by_month = train_df['DT_M']

del train_df


# In[ ]:


## Let's creat dataframes to compare results
## We will join prepdictions
RESULTS_1 = holdout_subset_1[[TARGET]]
RESULTS_2 = holdout_subset_2[[TARGET]]
RESULTS_3 = holdout_subset_3[[TARGET]]

all_results = {
        1: [RESULTS_1, holdout_subset_1, '    site_id holdout'],
        2: [RESULTS_2, holdout_subset_2, 'building_id holdout'],
        3: [RESULTS_3, holdout_subset_3, '      month holdout']
    }

for _,df in all_results.items():
    df[0]['test'] = 0    
    print('Ground RMSE for', df[2], '|',
          rmse(df[0][TARGET], df[0]['test']))
    del df[0]['test']
    print('#'*20)    
    
# We will always use same number of splits
# for training model
# Number of splits depends on data structure
# something in range 5-10
# 5 - is a common number of splits
# 10+ is too much (we will not have enough diversity in data)
# Here we will use 3 for faster training
# but you can change it by yourself
N_SPLITS = 3


# In[ ]:


# We don't know where to stop
# so we will try to guess 
# number of boosting rounds
for n_rounds in [25,50,100,200]:
    print('#'*20)
    print('No Validation training...', n_rounds, 'boosting rounds')
    corrected_lgb_params = lgb_params.copy()
    corrected_lgb_params['n_estimators'] = n_rounds
    corrected_lgb_params['early_stopping_rounds'] = None

    train_data = lgb.Dataset(X, label=y)
    
    estimator = lgb.train(
                corrected_lgb_params,
                train_data
            )

    for _,df in all_results.items():
        df[0]['no_validation_'+str(n_rounds)] = estimator.predict(df[1][features_columns])
        print('RMSE for',
              df[2], '|',
              rmse(df[0][TARGET], df[0]['no_validation_'+str(n_rounds)]))
        print('#'*20)

# Be careful. We are printing rmse results
# for our simulated test set
# but in real Data set we do not have True labels (obviously)
# and can't be sure that we stopped in right round
# lb probing can give you some idea how good our training is
# but this leads to nowhere -> overfits or completely bad results
# bad practice for real life problems!


# ### Findings
# 
# The main finding here is that we have "data leakage" in our dataset. And not single one.
# * Leakage by site_id -> our model doesn't generalize well for unkown site_id
# * Leakage by building_id -> our model doesn't generalize well for unkown building_id
# 
# What we can do here and do we have to do anything?
# 
# Good thing is all our test buildings and test sites present in train set.
# 
# Probably we don't need to smooth differences between them and can even make differences more explicit.
# 
# ---

# In[ ]:


print('#'*20)
print('KFold (with shuffle) training...')

from sklearn.model_selection import KFold
folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for _,df in all_results.items():
    df[0]['shuffle_kfold'] = 0
        
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print('Fold:',fold_+1)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    
    vl_x, v_y = X.iloc[val_idx,:], y[val_idx]    
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=v_y)  

    estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets = [train_data, valid_data],
            verbose_eval = 100,
        )
    
    for _,df in all_results.items():
        df[0]['shuffle_kfold'] += estimator.predict(df[1][features_columns])/N_SPLITS

for _,df in all_results.items():
    print('RMSE for', df[2], '|',
          rmse(df[0][TARGET], df[0]['shuffle_kfold']))
    print('#'*20)    


# In[ ]:


print('#'*20)
print('KFold (no shuffle) training...')

from sklearn.model_selection import KFold
folds = KFold(n_splits=N_SPLITS, shuffle=False, random_state=SEED)

for _,df in all_results.items():
    df[0]['no_shuffle_kfold'] = 0
        
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print('Fold:',fold_+1)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    
    vl_x, v_y = X.iloc[val_idx,:], y[val_idx]    
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=v_y)  

    estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets = [train_data, valid_data],
            verbose_eval = 100,
        )
    
    for _,df in all_results.items():
        df[0]['no_shuffle_kfold'] += estimator.predict(df[1][features_columns])/N_SPLITS

for _,df in all_results.items():
    print('RMSE for', df[2], '|',
          rmse(df[0][TARGET], df[0]['no_shuffle_kfold']))
    print('#'*20)    


# ### Findings
# 
# The main finding here is that we have one more "data leakage".
# * Leakage by date/month
# 
# Consumptions differ a lot month by month. 
# 
# We can't exclude any data by month as we need to predict consumptions for the whole year.
# 
# Our task becoming more and more interesting as we have to validate our features somehow.
# 
# We can't use normal kfold for validation because if the model knows how much energy was spent at 8 am it can make a good prediction for 9 am, but we don't have such data in our test set. 
# 
# ---

# In[ ]:


print('#'*20)
print('GroupKFold building_id split training...') 

from sklearn.model_selection import GroupKFold
folds = GroupKFold(n_splits=N_SPLITS)

for _,df in all_results.items():
    df[0]['Groupkfold_by_building'] = 0
      
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_by_building)):
    print('Fold:',fold_+1)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    
    vl_x, v_y = X.iloc[val_idx,:], y[val_idx]    
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=v_y)  

    estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets = [train_data, valid_data],
            verbose_eval = 100,
        )

    for _,df in all_results.items():
        df[0]['Groupkfold_by_building'] += estimator.predict(df[1][features_columns])/N_SPLITS

for _,df in all_results.items():
    print('RMSE for', df[2], '|',
          rmse(df[0][TARGET], df[0]['Groupkfold_by_building']))
    print('#'*20)  


# In[ ]:


print('#'*20)
print('GroupKFold site_id split training...') 

from sklearn.model_selection import GroupKFold
folds = GroupKFold(n_splits=N_SPLITS)

for _,df in all_results.items():
    df[0]['Groupkfold_by_site'] = 0
      
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_by_site)):
    print('Fold:',fold_+1)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    
    vl_x, v_y = X.iloc[val_idx,:], y[val_idx]    
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=v_y)  

    estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets = [train_data, valid_data],
            verbose_eval = 100,
        )

    for _,df in all_results.items():
        df[0]['Groupkfold_by_site'] += estimator.predict(df[1][features_columns])/N_SPLITS

for _,df in all_results.items():
    print('RMSE for', df[2], '|',
          rmse(df[0][TARGET], df[0]['Groupkfold_by_site']))
    print('#'*20)  


# In[ ]:


print('#'*20)
print('GroupKFold month split training...') 

from sklearn.model_selection import GroupKFold
folds = GroupKFold(n_splits=N_SPLITS)

for _,df in all_results.items():
    df[0]['Groupkfold_by_month'] = 0
      
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_by_month)):
    print('Fold:',fold_+1)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    
    vl_x, v_y = X.iloc[val_idx,:], y[val_idx]    
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=v_y)  

    estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets = [train_data, valid_data],
            verbose_eval = 100,
        )

    for _,df in all_results.items():
        df[0]['Groupkfold_by_month'] += estimator.predict(df[1][features_columns])/N_SPLITS

for _,df in all_results.items():
    print('RMSE for', df[2], '|',
          rmse(df[0][TARGET], df[0]['Groupkfold_by_month']))
    print('#'*20)  


# ### Findings
# 
# Same as before. "Leakage" prevents our model to generalize well.
# 
# ---

# ### Summary
# 
# For test set predictions our training set MUST have all building_ids and all months to make more accurate predictions.
# 
# 
# I would recommend trying train/skip/validate for feature validation:
# 
# * Train set - first 4 month
# * Skip - next 4 month
# * Valid set - last 4 month
# 
# 
# For test set predictions use slightly more boosting rounds than validation scheme early stopping will show.
# 
# Train several seed models (not kfold, just different seed).
# 
# Average results.
# 
# ---
