#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, gc, warnings, random, math
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')
data_folder = "../input/ieee-fraud-detection/"


# In[ ]:



print('Reading data...')

train_identity = pd.read_csv(f'{data_folder}train_identity.csv', index_col='TransactionID')
train_transaction = pd.read_csv(f'{data_folder}train_transaction.csv', index_col='TransactionID')
test_identity = pd.read_csv(f'{data_folder}test_identity.csv', index_col='TransactionID')
test_transaction = pd.read_csv(f'{data_folder}test_transaction.csv', index_col='TransactionID')
sub = pd.read_csv(f'{data_folder}sample_submission.csv')

print('Merging data...')
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
print("Done")

del train_identity, train_transaction, test_identity, test_transaction
gc.collect()

test['isFraud'] = -1


# In[ ]:



# Generating day of first transaction from D1
for df in [train, test]:
    for col in ['D1']:
        df[col+'_shift'] = (df[col] - df.TransactionDT // 24 // 3600).fillna(-400)


# In[ ]:



# First userid based on existing and generated features
for df in [train, test]:
    df['uid'] = df['D1_shift'].astype(str)+'_'+df['card1'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['ProductCD'].astype(str)


# In[ ]:


START_DATE = datetime.strptime('2017-11-30', '%Y-%m-%d')
train['DT'] = train['TransactionDT'].apply(lambda x: (START_DATE + timedelta(seconds = x)))
test['DT'] = test['TransactionDT'].apply(lambda x: (START_DATE + timedelta(seconds = x)))

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

# The function tried to split users having the same uid, cheking the following assumptions:
# - D2 is monotonic for a user
# V307 is a cumulative sum of TransactionAmt over previous 30 days
# V308 is a cumulative sum of TransactionAmt over previous 7 days
# V306 is a cumulative sum of TransactionAmt over previous 24 hours
def cal_sub_id(df_base, eps = 1e-2):
    
    df = df_base.copy()
    df.reset_index(level=0, inplace=True)
    df['sub_id'] = -1

    sub_id = 1
    start_i = 0
    
    while start_i < len(df):
        if df.loc[start_i, 'sub_id'] > 0:
            start_i += 1
            continue
        df.loc[start_i, 'sub_id'] = sub_id
        i1 = start_i
        for i2 in range(i1+1, len(df)):
            if df.loc[i2, 'sub_id'] > 0:
                continue
            cond = (df.index < i2) & (df['sub_id'] == sub_id)
            sum_month = df.loc[(df['DT'] >= df.loc[i2, 'DT'] - timedelta(days=30) ) & cond, "TransactionAmt"].sum()
            sum_week  = df.loc[(df['DT'] >= df.loc[i2, 'DT'] - timedelta(days=7)  ) & cond, "TransactionAmt"].sum()
            sum_day =   df.loc[(df['DT'] >= df.loc[i2, 'DT'] - timedelta(hours=24)) & cond, "TransactionAmt"].sum()
            
            if ((abs(df.loc[i2, 'D1'] - df.loc[i1, 'D1'] - df.loc[i2, 'D3']) < 2 or np.isnan(df.loc[i2, 'D3'])) and                 ((df.loc[i2, 'D2'] >= df.loc[i1, 'D2']) or np.isnan(df.loc[i2, 'D2']) or np.isnan(df.loc[i1, 'D2'])) and                 ((df.loc[i2, 'V307'] <= eps + df.loc[i1, 'V307'] + df.loc[i1, 'TransactionAmt'] and df.loc[i2, 'V307'] >= sum_month - eps) or np.isnan(df.loc[i2, 'V307']) or np.isnan(df.loc[i1, 'V307']) ) and                 ((df.loc[i2, 'V308'] <= eps + df.loc[i1, 'V308'] + df.loc[i1, 'TransactionAmt'] and df.loc[i2, 'V308'] >= sum_week - eps) or np.isnan(df.loc[i2, 'V308']) or np.isnan(df.loc[i1, 'V308']) ) and                 ((df.loc[i2, 'V306'] <= eps + df.loc[i1, 'V306'] + df.loc[i1, 'TransactionAmt'] and df.loc[i2, 'V306'] >= sum_day - eps) or np.isnan(df.loc[i2, 'V306']) or np.isnan(df.loc[i1, 'V306']) )):
                i1 = i2
                df.loc[i1, 'sub_id'] = sub_id
        sub_id += 1

    return pd.Series(df['sub_id'].values, index=df_base.index)


# In[ ]:


# Improving uid, saving to uid2

cols = ['DT', 'isFraud', 'ProductCD', 'D1', 'D2', 'D3', 'D15', 'V306', 'V307', 'V308', 'TransactionAmt', 'uid']

start_time = time.time()
train2 = train[cols].copy()
train2 = pd.merge(train2, applyParallel(train2.groupby(['uid']), cal_sub_id).rename('sub_id'), on=['TransactionID'], how='inner')
print(time.time() - start_time)


start_time = time.time()
test2 = test[cols].copy()
test2 = pd.merge(test2, applyParallel(test2.groupby(['uid']), cal_sub_id).rename('sub_id'), on=['TransactionID'], how='inner')
print(time.time() - start_time)


for df in [train2, test2]:
    df['uid2'] = df['uid'].astype(str) + '_' + df['sub_id'].astype(str)


# In[ ]:


# The same uid contains multiple uid2 values in train and test.
# The following lazy wirtten function tries to map uid2 with most similar medians of TransactionAmt

def closest_val(d, v):
    res_key = None
    res_dist = -1
    for d_key, d_val in d.items():
        dst = abs(d_val - v)
        if res_key is None or dst < res_dist:
            res_key = d_key
            res_dist = dst
    return res_key

def map_ids(df):
    train_df = df[df['isFraud'] != -1]
    test_df  = df[df['isFraud'] == -1]
    
    res = {}
    
    test_match_ids = []
    test_new_ids = []
    for i in test_df['uid2'].unique():
        test_match_ids.append(i)
    
    train_avg = train_df.groupby('uid2')['TransactionAmt'].median().to_dict()
    test_avg  = test_df[test_df['uid2'].isin(test_match_ids)].groupby('uid2')['TransactionAmt'].median().to_dict()
    
    for k, v in test_avg.items():
        if len(train_avg) == 0:
            test_new_ids.append(k)
            continue
        
        tr_key = closest_val(train_avg, v)
        res[k] = tr_key
        del train_avg[tr_key]
        
    for i, k in enumerate(test_new_ids):
        res[k] = k + '_' + f"x{i}"
        
    temp = df[['isFraud', 'uid2']].copy()
    temp.loc[temp['isFraud'] == -1, 'uid2'] = temp.loc[temp['isFraud'] == -1, 'uid2'].map(res)
    
    return temp['uid2']


# In[ ]:


# Generating uid3, based on uid2 and applying the matching algorithm from above

cols = ['DT', 'isFraud', 'ProductCD', 'D1', 'D3', 'D15', 'V306', 'V307', 'V308', 'TransactionAmt', 'uid', 'sub_id', 'uid2']

start_time = time.time()
temp = train2.append(test2)[cols].copy()
temp = pd.merge(temp, applyParallel(temp.groupby(['uid']), map_ids).rename('uid3'), on=['TransactionID'], how='inner')
print(time.time() - start_time)

train2['uid3'] = temp['uid3'][:len(train2)].values
test2['uid3'] = temp['uid3'][len(train2):].values


# In[ ]:


cols = ['DT', 'uid3']

train2[cols].to_csv("train_ids.csv")
test2[cols].to_csv("test_ids.csv")


# In[ ]:




