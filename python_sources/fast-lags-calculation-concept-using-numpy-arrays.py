#!/usr/bin/env python
# coding: utf-8

# This kernel shows how to prepare lags separately for train and test phases

# In[ ]:


# Locally I created classes with a similar behaviour that we have in this competition
# It slowed me down during validation phase and I decided to refactor code for lags creation

# from twosigmanews import *
from kaggle.competitions import twosigmanews


# In[ ]:


import pandas as pd
import numpy as np
import gc
from resource import getrusage, RUSAGE_SELF
from datetime import date, datetime

import multiprocessing
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


global STARTED_TIME
STARTED_TIME = datetime.now()

# It's better to use cpu_count from the system - who knows what happens during test phase
global N_THREADS
N_THREADS=multiprocessing.cpu_count()

print(f'N_THREADS: {N_THREADS}')


# In[ ]:


# FILTERDATE - start date for the train data
FILTERDATE = date(2007, 1, 1)

# SAMPLEDATE - I use it for sampling and fast sanity check of scripts
SAMPLEDATE = None
# SAMPLEDATE = date(2007, 1, 30)


# In[ ]:


global N_LAG, RETURN_FEATURES

# Let's try how it works for 1-year lags
N_LAG = np.sort([5, 10, 20, 252])

# Features for lags calculation
RETURN_FEATURES = [
    'returnsOpenPrevMktres10',
    'returnsOpenPrevRaw10',
    'open',
    'close']


# In[ ]:


# Tracking time and memory usage
global MAXRSS
MAXRSS = getrusage(RUSAGE_SELF).ru_maxrss
def using(point=""):
    global MAXRSS, STARTED_TIME
    print(str(datetime.now()-STARTED_TIME).split('.')[0], point, end=' ')
    max_rss = getrusage(RUSAGE_SELF).ru_maxrss
    if max_rss > MAXRSS:
        MAXRSS = max_rss
    print(f'max RSS {MAXRSS/1024/1024:.1f}Gib')
    gc.collect();


# In[ ]:


# I've slightly optimized this function from https://www.kaggle.com/qqgeogor/eda-script-67
# It uses just one loop ofr N_LAG

global FILLNA
FILLNA = -1

def create_lag(df_code):
    prevlag = 1    
    for window in N_LAG:
        rolled = df_code[RETURN_FEATURES].shift(prevlag).rolling(window=window)
        # Mean is not so stable as median if you have assets with very high/low beta risk factor
        # df_code = df_code.join(rolled.mean().add_suffix(f'_lag_{window}_mean'))
        df_code = df_code.join(rolled.median().add_suffix(f'_lag_{window}_median'))
        df_code = df_code.join(rolled.max().add_suffix(f'_lag_{window}_max'))
        df_code = df_code.join(rolled.min().add_suffix(f'_lag_{window}_min'))

        # We also have an idea to make lags uncorrelated - in this case you may uncomment
        # the following line. N_LAG have to be sorted

        # prevlag = window
    return df_code.fillna(FILLNA)

def generate_lag_features(df):
    global RETURN_FEATURES, N_THREADS
    all_df = []
    df_codes = df.groupby('assetCode')
    df_codes = [df_code[1][['time','assetCode']+RETURN_FEATURES] for df_code in df_codes]
    
    pool = Pool(N_THREADS)
    all_df = pool.map(create_lag, df_codes)
    
    new_df = pd.concat(all_df)  
    new_df.drop(RETURN_FEATURES,axis=1,inplace=True)
    pool.close()
    
    return new_df


# In[ ]:


# Pre-processing of dataframe, this functions is the same for train and test periods
# In production we had more calculations

def preparedf(df):
    df['time'] = df['time'].dt.date
    return df


# In[ ]:


# The following functions are used for initialization and expanding of numpy arrays
# for storing historical data of all assets.

# It helps to have very fast lags creation.

# Initialization of history array
def initialize_values(items=5000, features=4, history=15):
    return np.ones((items, features, history))*np.nan

# Expanding of history array for new assets
def add_values(a, items=100):
    return np.concatenate([a, initialize_values(items, a.shape[1], a.shape[2])])

# codes dictionary maps assetCode to the index in the history array
# if we found new code - we have to store it and expand history
def get_code(a):
    global codes, history
    try: 
        return codes[a]
    except KeyError:
        codes[a] = len(codes)
        if len(codes) > history.shape[0]:
            history = add_values(history, 100)
        return codes[a]

# list2codes returns numpy array of indexes of assetCodes (for each day)
def list2codes(l):
    return np.array([get_code(a) for a in l])


# Let's start

# In[ ]:


env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
using('Done')


# In[ ]:


print('Dataframe pre-processing')
market_train_df = preparedf(market_train_df)
using('Done')


# In[ ]:


# Dataframe filtering
print('DF Filtering')
market_train_df = market_train_df.loc[market_train_df['time']>=FILTERDATE]

if SAMPLEDATE is not None:
    market_train_df = market_train_df.loc[market_train_df['time']<=SAMPLEDATE]  
using('Done')


# In[ ]:


print('Lag features generation')
new_df = generate_lag_features(market_train_df)
using('Done')


# In[ ]:


print('DF Merging')
market_train_df = pd.merge(market_train_df,new_df,how='left',on=['time','assetCode'])
del new_df
using('Done')


# In[ ]:


print('Preparation for the prediction')
# codes maps assetCodes with indexes into history array
codes = dict(
    zip(market_train_df.assetCode.unique(), np.arange(market_train_df.assetCode.nunique()))
)
# history stores information for all assets, required features
# np.max(LAG)+1 - to store information for maximum beriod and the current day (+1)
history = initialize_values(len(codes), len(RETURN_FEATURES), np.max(N_LAG)+1)

# Get the latest information for assets
latest_events = market_train_df.groupby('assetCode').tail(np.max(N_LAG)+1)
# but we may have different informations size for different assets
latest_events_size = latest_events.groupby('assetCode').size()

# Filling the history array
for s in latest_events_size.unique():
    for i in range(len(RETURN_FEATURES)):
        # l is a Dataframe with assets with the same history size for each asset
        l = latest_events[
            latest_events.assetCode.isin(latest_events_size[latest_events_size==s].index.values)
        ].groupby('assetCode')[RETURN_FEATURES[i]].apply(list)

        # v is a 2D array contains history information of feature RETURN_FEATURES[i] 
        v = np.array([k for k in l.values])

        # r contains indexes (in the history array) of all assets
        r = list2codes(l.index.values)

        # Finally, filling history array
        history[r, i, -s:] = v
        del l, v, r

del latest_events, latest_events_size
using('Done')


# In[ ]:


print('Prediction')
#prediction
days = env.get_prediction_days()
n_days = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if n_days % 100 == 0:
        using(f'{n_days}')
    # Test data preprocessing    
    market_obs_df = preparedf(market_obs_df)

    # Getting indexes of asses for the current day
    r = list2codes(market_obs_df.assetCode.values)

    # Shifting history by 1 for assets of the current day
    history[r, :, :-1] = history[r, :, 1:] 

    # Filling history with a new data
    history[r, :, -1] = market_obs_df[RETURN_FEATURES].values

    prevlag = 1    
    for lag in np.sort(N_LAG):
        lag_median = np.median(history[r, : , -lag:-prevlag], axis=2)
        lag_median[np.isnan(lag_median)] = FILLNA
        lag_max = history[r, : , -lag-1:-prevlag].max(axis=2)
        lag_max[np.isnan(lag_max)] = FILLNA
        lag_min = history[r, : , -lag-1:-prevlag].min(axis=2)
        lag_min[np.isnan(lag_min)] = FILLNA

        for ix in range(len(RETURN_FEATURES)):
            market_obs_df[f'{RETURN_FEATURES[ix]}_lag_{lag}_median'] = lag_median[:, ix]
            market_obs_df[f'{RETURN_FEATURES[ix]}_lag_{lag}_min'] = lag_min[:, ix]
            market_obs_df[f'{RETURN_FEATURES[ix]}_lag_{lag}_max'] = lag_max[:, ix]
            
#         prevlag=lag

    confidence = 0
    
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})

    predictions_template_df = predictions_template_df.merge(preds,how='left')    .drop('confidenceValue',axis=1)    .fillna(0)    .rename(columns={'confidence':'confidenceValue'})
    
    env.predict(predictions_template_df)
    
using('Prediction done')

# env.write_submission_file()
# using('Done')


# In[ ]:




