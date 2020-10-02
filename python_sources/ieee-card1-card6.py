#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime, math

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

from scipy.stats import ks_2samp

import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# In[ ]:


########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


########################### Vars
#################################################################################
SEED = 42
seed_everything(SEED)
LOCAL_TEST = True
MAKE_MODEL_TEST = True
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')


# In[ ]:


########################### Model params
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
                    'n_estimators':80000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                } 


# In[ ]:


########################### Model
import lightgbm as lgb

def make_test_predictions(tr_df, tt_df, target, lgb_params, NFOLDS=2):
    
    new_columns = set(list(train_df)).difference(base_columns + remove_features)
    features_columns = base_columns + list(new_columns)
    
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  

    for col in list(X):
        if X[col].dtype=='O':
            X[col] = X[col].fillna('unseen_before_label')
            P[col] = P[col].fillna('unseen_before_label')

            X[col] = train_df[col].astype(str)
            P[col] = test_df[col].astype(str)

            le = LabelEncoder()
            le.fit(list(X[col])+list(P[col]))
            X[col] = le.transform(X[col])
            P[col]  = le.transform(P[col])

            X[col] = X[col].astype('category')
            P[col] = P[col].astype('category')
        
    tt_df = tt_df[['TransactionID',target]]    
    predictions = np.zeros(len(tt_df))

    tr_data = lgb.Dataset(X, label=y)
    vl_data = lgb.Dataset(P, label=P_y) 
    estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets = [tr_data, vl_data],
            verbose_eval = 200,
        )   
        
    pp_p = estimator.predict(P)
    predictions += pp_p/NFOLDS

    if LOCAL_TEST:
        feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])
        print(feature_imp)
        
    tt_df['prediction'] = predictions
    
    return tt_df
## -------------------


# ----

# In[ ]:


########################### DATA LOAD
#################################################################################
print('Load Data')
train_df = pd.read_pickle('../input/ieee-data-minification/train_transaction.pkl')

if LOCAL_TEST:
    
    # Convert TransactionDT to "Month" time-period. 
    # We will also drop penultimate block 
    # to "simulate" test set values difference
    train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month 
    test_df = train_df[train_df['DT_M']==train_df['DT_M'].max()].reset_index(drop=True)
    train_df = train_df[train_df['DT_M']<(train_df['DT_M'].max()-1)].reset_index(drop=True)
    del train_df['DT_M'], test_df['DT_M']
    
else:
    test_df = pd.read_pickle('../input/ieee-data-minification/test_transaction.pkl')
    
print('Shape control:', train_df.shape, test_df.shape)


# In[ ]:


########################### Features
#################################################################################
# Add list of feature that we will
# remove later from final features list
remove_features = [
    'TransactionID','TransactionDT', # These columns are pure noise right now
    TARGET,
    ]

# Let's also remove all V columns for tests
remove_features += ['V'+str(i) for i in range(1,340)]

base_columns = [col for col in list(train_df) if col not in remove_features]


# In[ ]:


#### Let's make baseline model 
if MAKE_MODEL_TEST:
    test_predictions = make_test_predictions(train_df, test_df, TARGET, lgb_params)
    print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
####


# In[ ]:


########################### Let's check how many nans we have
i_cols = ['card1','card2','card3','card4','card5','card6']

for col in i_cols:
    print(col,':',train_df[col].isna().sum())
    
train_df[i_cols].head()


# In[ ]:


########################### Let's play "sudoku" and fill nans in cards columns
i_cols = ['TransactionID','card1','card2','card3','card4','card5','card6']

full_df = pd.concat([train_df[i_cols], test_df[i_cols]])

## I've used frequency encoding before so we have ints here
## we will drop very rare cards
full_df['card6'] = np.where(full_df['card6']==30, np.nan, full_df['card6'])
full_df['card6'] = np.where(full_df['card6']==16, np.nan, full_df['card6'])

i_cols = ['card2','card3','card4','card5','card6']

## We will find best match for nan values and fill with it
for col in i_cols:
    temp_df = full_df.groupby(['card1',col])[col].agg(['count']).reset_index()
    temp_df = temp_df.sort_values(by=['card1','count'], ascending=False).reset_index(drop=True)
    del temp_df['count']
    temp_df = temp_df.drop_duplicates(keep='first').reset_index(drop=True)
    temp_df.index = temp_df['card1'].values
    temp_df = temp_df[col].to_dict()
    full_df[col] = np.where(full_df[col].isna(), full_df['card1'].map(temp_df), full_df[col])
    
    
i_cols = ['card1','card2','card3','card4','card5','card6']
for col in i_cols:
    train_df[col] = full_df[full_df['TransactionID'].isin(train_df['TransactionID'])][col].values
    test_df[col] = full_df[full_df['TransactionID'].isin(test_df['TransactionID'])][col].values


# In[ ]:


########################### Let's check how many nans left
i_cols = ['card1','card2','card3','card4','card5','card6']

for col in i_cols:
    print(col,':',train_df[col].isna().sum())
    
train_df[i_cols].head()


# In[ ]:


#### Let's find out if we have boost
if MAKE_MODEL_TEST:
    test_predictions = make_test_predictions(train_df, test_df, TARGET, lgb_params)
    print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
####


# ----
# 
# ## NOTES!
# 
# 1. Can you do it better? Yes? Look for combinations in cards columns and make more specific fillup.
# 
# 2. Does it help? For me yes. Please trust your CV and not my word.
# 
# 3. Should it be done before all fe or after -> trust your cv
# 
# 4. This is just an example.
# 
