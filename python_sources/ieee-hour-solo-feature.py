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
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')


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


########################### We need some time features for our experiments
for df in [train_df, test_df]:
    
    # Temporary variables for aggregation
    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)
    df['DT_W'] = ((df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear).astype(np.int8)
    df['DT_D'] = ((df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear).astype(np.int16)
    
    df['DT_hour'] = (df['DT'].dt.hour).astype(np.int8)
    df['DT_day_week'] = (df['DT'].dt.dayofweek).astype(np.int8)
    df['DT_day_month'] = (df['DT'].dt.day).astype(np.int8)

# Total transactions per timeblock
for col in ['DT_M','DT_W','DT_D']:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()
            
    train_df[col+'_total'] = train_df[col].map(fq_encode)
    test_df[col+'_total']  = test_df[col].map(fq_encode)


# In[ ]:


########################### I would like to check DT_hour and D9 (that is also hour of transaction)
test_group = ['DT_hour','D9']
periods = ['TransactionDT']
temp_df = pd.concat([train_df[test_group+periods], test_df[test_group+periods]])
for period in periods:
    for col in test_group:
        for df in [temp_df]:
            df.set_index(period)[col].plot(style='.', title=col, figsize=(15, 3))
            plt.show()
            
# Seems ok
# What mean nans in D9 - I don't know
# Is it Server time or local time - I don't know
# But it doesn't metter for us now


# In[ ]:


########################### We wanted to check mean hour for normal transactions and for fraud
test_group = ['TransactionID','TransactionDT','DT_hour','D9','DT_M','DT_W','DT_D','isFraud']

df = pd.concat([train_df[test_group], test_df[test_group]])

df_fraud = df[df['isFraud']==1]
df_not_fraud = df[df['isFraud']==0]

print('#'*10)
print('Mean hour by fraud group')
print('Fraud mean hour', df_fraud['DT_hour'].mean())
print('Not Fraud mean hour', df_not_fraud['DT_hour'].mean())

print('#'*10)
print('Mean fraud by hour')
print(df.groupby(['DT_hour'])['isFraud'].agg(['mean']).sort_values(by='mean', ascending=False))

print('#'*10)
print('Mean fraud:')


cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
for time_block in ['DT_M','DT_W','DT_D']:
    plt.figure(figsize=(16, 6))
    for period in list(df[time_block].unique()):
        data = df[df[time_block]==period].groupby(['DT_hour'])['isFraud'].agg(['mean']).reset_index()
        ax = sns.scatterplot(x="DT_hour", y="mean", palette=cmap,data=data)            
    plt.show()


# 
