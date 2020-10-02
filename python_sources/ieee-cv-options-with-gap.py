#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import lightgbm as lgb

import math
warnings.filterwarnings('ignore')


# In[ ]:


########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


# In[ ]:


########################### Vars
#################################################################################
SEED = 42
seed_everything(SEED)
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')


# In[ ]:


########################### Model params
# These parameters we will keep untouched
# for each lgbm model
# the unique param that we will look at
# is n_estimators
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'n_estimators':20000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                } 


# In[ ]:


########################### DATA LOAD
#################################################################################
print('Load Data')
train_df = pd.read_pickle('../input/ieee-data-minification/train_transaction.pkl')

# We will prepare simulation here
# Last month will be our test test 
train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month 

test_df = train_df[train_df['DT_M']==train_df['DT_M'].max()].reset_index(drop=True)
train_df = train_df[train_df['DT_M']<(train_df['DT_M'].max()-1)].reset_index(drop=True)
    
print('Shape control:', train_df.shape, test_df.shape)


# In[ ]:


########################### Encode Str columns
# For all such columns (probably not)
# we already did frequency encoding (numeric feature)
# so we will use astype('category') here
for col in list(train_df):
    if train_df[col].dtype=='O':
        print(col)
        train_df[col] = train_df[col].fillna('unseen_before_label')
        test_df[col]  = test_df[col].fillna('unseen_before_label')
        
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        
        le = LabelEncoder()
        le.fit(list(train_df[col])+list(test_df[col]))
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
        
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')


# In[ ]:


########################### Model Features 
# Remove Some Features
rm_cols = [
    'TransactionID','TransactionDT', # These columns are pure noise right now
    TARGET,                          # Not target in features))
    'DT_M'                           # Column that we used to simulate test set
]

# Remove V columns (for faster training)
rm_cols += ['V'+str(i) for i in range(1,340)]

# Final features
features_columns = [col for col in list(train_df) if col not in rm_cols]

## Baseline LB score is 0.9360


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
# 2. Define Validation Metric (in our case it is ROC-AUC)
# 3. Stop training when Validation metric stops improving
# 4. Make predictions for Test set
# 
# Seems simple but he devil's always in the details.

# In[ ]:


## Let's creat dataframe to compare results
## We will join prepdictions
RESULTS = test_df[['TransactionID',TARGET]]

# We will always use same number of splits
# for training model
# Number of splits depends on data structure
# and in our case it is better to use 
# something in range 5-10
# 5 - is a common number of splits
# 10+ is too much (we will not have enough diversity in data)
# Here we will use 3 for faster training
# but you can change it by yourself
N_SPLITS = 3


# ----
# ## No Validation

# In[ ]:


# Main Data
# We will take whole train data set
# and will NOT use any early stopping 
X,y = train_df[features_columns], train_df[TARGET]

# Test Data (what we need to predict)
P = test_df[features_columns]

# We don't know where to stop
# so we will try to guess 
# number of boosting rounds
for n_rounds in [500,1000,2500,5000]:
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

    RESULTS['no_validation_'+str(n_rounds)] = estimator.predict(P)
    print('AUC score', metrics.roc_auc_score(RESULTS[TARGET], RESULTS['no_validation_'+str(n_rounds)]))
    print('#'*20)

# Be careful. We are printing auc results
# for our simulated test set
# but in real Data set we do not have True labels (obviously)
# and can't be sure that we stopped in right round
# lb probing can give you some idea how good our training is
# but this leads to nowhere -> overfits or completely bad results
# bad practice for real life problems!


# ----
# ## Kfold

# In[ ]:


print('#'*20)
print('KFold training...')

# You can find oof name for this strategy
# oof - Out Of Fold
# as we will use one fold as validation
# and stop training when validation metric
# stops improve
from sklearn.model_selection import KFold
folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Main Data
X,y = train_df[features_columns], train_df[TARGET]

# Test Data
P = test_df[features_columns]
RESULTS['kfold'] = 0

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
            verbose_eval = 1000,
        )

    RESULTS['kfold'] = estimator.predict(P)

print('AUC score', metrics.roc_auc_score(RESULTS[TARGET], RESULTS['kfold']))
print('#'*20)
    
## We have two "problems" here
## 1st: Training score goes upto 1 and it's not normal situation
## It's nomally means that model did perfect or
## almost perfect match between "data fingerprint" and target
## we definitely should stop before to generalize better
## 2nd: Our LB probing gave 0.936 and it is too far away from validation score
## some difference is normal, but such gap is too big


# ----
# ## StratifiedKFold
# There are situations when normal kfold split doesn't perform well because of train set imbalance.
# We can use StratifiedKFold to garant that each split will have same number of positives and negatives samples.

# In[ ]:


print('#'*20)
print('StratifiedKFold training...')

# Same as normal kfold but we can be sure
# that our target is perfectly distribuited
# over folds
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Main Data
X,y = train_df[features_columns], train_df[TARGET]

# Test Data and expport DF
P = test_df[features_columns]
RESULTS['stratifiedkfold'] = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=y)):
    print('Fold:',fold_+1)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    
    vl_x, v_y = X.iloc[val_idx,:], y[val_idx]    
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=v_y)  

    estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets = [train_data, valid_data],
            verbose_eval = 1000,
        )

    # we are not sure what fold is best for us
    # so we will average prediction results 
    # over folds
    RESULTS['stratifiedkfold'] += estimator.predict(P)/N_SPLITS

print('AUC score', metrics.roc_auc_score(RESULTS[TARGET], RESULTS['stratifiedkfold']))
print('#'*20)

## We have same "problems" here as in normal kfold
## 1st: Training score goes upto 1 and it's not normal situation
## we definitely should stop before 
## 2nd: Our LB probing gave 0.936 and it is too far away from validation score
## some difference is normal, but such gap is too big


# ----
# ## LBO (last block out)
# For Time series data (what we have here) we can use (sometimes) last Time block as validation subset and track mean early stopping round.
# 
# Let's code it.

# In[ ]:


print('#'*20)
print('LBO training...') 

## We need Divide Train Set by Time blocks
## Convert TransactionDT to Months
## And use last month as Validation
train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month 

main_train_set = train_df[train_df['DT_M']<(train_df['DT_M'].max())].reset_index(drop=True)
validation_set = train_df[train_df['DT_M']==train_df['DT_M'].max()].reset_index(drop=True)

## We will use oof kfold to find "best round"
folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Main Data
X,y = main_train_set[features_columns], main_train_set[TARGET]

# Validation Data
v_X, v_y = validation_set[features_columns], validation_set[TARGET]

estimators_bestround = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print('Fold:',fold_+1)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(v_X, label=v_y)  

    seed_everything(SEED)
    estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets = [train_data, valid_data],
            verbose_eval = 1000,
        )
    estimators_bestround.append(estimator.current_iteration())

## Now we have "mean Best round" and we can train model on full set
corrected_lgb_params = lgb_params.copy()
corrected_lgb_params['n_estimators'] = int(np.mean(estimators_bestround))
corrected_lgb_params['early_stopping_rounds'] = None
print('#'*10)
print('Mean Best round:', corrected_lgb_params['n_estimators'])

# Main Data
X,y = train_df[features_columns], train_df[TARGET]

# Test Data
P = test_df[features_columns]
RESULTS['lbo'] = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print('Fold:',fold_+1)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
    train_data = lgb.Dataset(tr_x, label=tr_y)

    estimator = lgb.train(
            corrected_lgb_params,
            train_data
        )
    
    RESULTS['lbo'] += estimator.predict(P)/N_SPLITS

print('AUC score', metrics.roc_auc_score(RESULTS[TARGET], RESULTS['lbo']))
print('#'*20)   


# ----
# ## GroupKFold
# > The folds are approximately balanced in the sense that the number of distinct groups is approximately the same in each fold.
# 
# Why we may use it?
# Let's imagine that we want to separate train data by time blocks groups or client IDs or something else.
# With GroupKFold we can be sure that our validation fold will contain groupIDs that are not in main train set.
# Sometimes it helps to deal with "dataleakage" and overfit.

# In[ ]:


print('#'*20)
print('GroupKFold timeblocks split training...') 

from sklearn.model_selection import GroupKFold
folds = GroupKFold(n_splits=N_SPLITS)

## We need Divide Train Set by Time blocks
## Convert TransactionDT to Months
train_df['groups'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train_df['groups'] = (train_df['groups'].dt.year-2017)*12 + train_df['groups'].dt.month 

# Main Data
X,y = train_df[features_columns], train_df[TARGET]
split_groups = train_df['groups']

# Test Data and expport DF
P = test_df[features_columns]
RESULTS['groupkfold_timeblocks'] = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
    print('Fold:',fold_+1)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    
    vl_x, v_y = X.iloc[val_idx,:], y[val_idx]    
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=v_y)  

    estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets = [train_data, valid_data],
            verbose_eval = 1000,
        )

    RESULTS['groupkfold_timeblocks'] += estimator.predict(P)/N_SPLITS

print('AUC score', metrics.roc_auc_score(RESULTS[TARGET], RESULTS['groupkfold_timeblocks']))
print('#'*20)


# In[ ]:


print('#'*20)
print('GroupKFold uID split training...') 

from sklearn.model_selection import GroupKFold
folds = GroupKFold(n_splits=N_SPLITS)

## We need Divide Train Set by virtual client ID
## If we do everuthing well
## (I'm not sure that this columns are good ones
## 'card1','card2','card3','card5','addr1','addr2')
## our model will not have "personal client information"
## shared by folds

train_df['groups'] = ''
for col in ['card1','card2','card3','card5','addr1','addr2',]:
    train_df['groups'] = '_' + train_df[col].astype(str)
    
# Main Data
X,y = train_df[features_columns], train_df[TARGET]
split_groups = train_df['groups']

# Test Data and expport DF
P = test_df[features_columns]
RESULTS['groupkfold_uid'] = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
    print('Fold:',fold_+1)
    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    
    vl_x, v_y = X.iloc[val_idx,:], y[val_idx]    
    train_data = lgb.Dataset(tr_x, label=tr_y)
    valid_data = lgb.Dataset(vl_x, label=v_y)  

    estimator = lgb.train(
            lgb_params,
            train_data,
            valid_sets = [train_data, valid_data],
            verbose_eval = 1000,
        )

    RESULTS['groupkfold_uid'] += estimator.predict(P)/N_SPLITS

print('AUC score', metrics.roc_auc_score(RESULTS[TARGET], RESULTS['groupkfold_uid']))
print('#'*20)


# In[ ]:





# In[ ]:


print('#'*30)
print('Final results...')
final_df = []
for current_strategy in list(RESULTS.iloc[:,2:]):
    auc_score = metrics.roc_auc_score(RESULTS[TARGET], RESULTS[current_strategy])
    final_df.append([current_strategy, auc_score])
    
final_df = pd.DataFrame(final_df, columns=['Stategy', 'Result'])
final_df.sort_values(by=['Result'], ascending=False, inplace=True)
print(final_df)

