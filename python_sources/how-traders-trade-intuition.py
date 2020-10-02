#!/usr/bin/env python
# coding: utf-8

# This kernel will describe basic trading strategies used by stock traders in order to give intuition on manual trading.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews
import gc
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


env = twosigmanews.make_env()
(train_market_data, _) = env.get_training_data()


# In[ ]:


#get sample asset
test_asset = train_market_data.loc[train_market_data['assetCode']=='A.N'].sort_values('time').tail(200).reset_index(drop=True)


# **Relative Signal Strength**
# 
# RSI is used to indicate the trend momentum
# The upper region is called overbought and the lower region is oversold
# Once the RSI reaches the overbought or oversold region it means the trend is strongly going down or up respectively, but once it exits the region then a trend reversal might occur.

# In[ ]:


def calcRsi(series, period):
    """
    Calculate the RSI of a data series 
    
    Parameters
    ----------
    series : pandas series
        Candle sticks dataset
    period : int
        Period of each calculation
        
    Returns
    -------
    rsi : float
        the calculated rsi
    """
    try:
        delta = series.diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
        u = u.drop(u.index[:(period-1)])
        d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
        d = d.drop(d.index[:(period-1)])

        rs = u.ewm(com=period-1, adjust=False).mean()             / d.ewm(com=period-1, adjust=False).mean()
        
        rsi = 100 - 100 / (1 + rs)
    except IndexError:
        rsi = 0
        
    return rsi
test_asset['RSI'] = calcRsi(test_asset['close'], 14)
#RSI
fig, ax = plt.subplots()
ax.plot(test_asset['time'], test_asset['RSI'])
ax.axhline(y=70,color='r')
ax.axhline(y=30,color='r')
plt.text(s='Overbought', x=test_asset['time'].iloc[0], y=70, fontsize=14)
plt.text(s='OverSold', x=test_asset['time'].iloc[0], y=30, fontsize=14)
plt.legend()
p = plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, fontsize=8)


# **Bollinger Bands**
# 
# BB is used to indicate possible trend reversals. Once the trendline crosses any of the upper or lower band then it means a reversal may occur

# In[ ]:


def addBollinger(df, period=20, col='close'):
    """
    Add the simple moving average column to dataframe 

    Parameters
    ----------
    df : pandas dataframe
        Candle sticks dataset
    period : int
        Period of each calculation

    Returns
    -------
    None
    """
    bbmid_series = df[col].rolling(window=period).mean()
    series_stdev = df[col].rolling(window=period).std()
    df['BBUpperBand'] = bbmid_series + 2*series_stdev
    df['BBLowerBand'] = bbmid_series - 2*series_stdev
    df['BBBandwidth'] = df['BBUpperBand'] - df['BBLowerBand']  
    df['BBMiddleBand'] = bbmid_series
    return df

test_asset = addBollinger(test_asset)
#Bollinger Bands
fig, ax = plt.subplots()
ax.plot(test_asset['time'], test_asset['close'])
ax.plot(test_asset['time'], test_asset['BBUpperBand'], c='orange')
ax.plot(test_asset['time'], test_asset['BBLowerBand'], c='orange')
ax.plot(test_asset['time'], test_asset['BBMiddleBand'], c='black')
plt.legend()
p = plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, fontsize=8)


# **Moving Average Convergence Divergence**
# 
# MACD is another momentum indicator that helps in telling if the market is losing steam in its current trend and reversal may occur. Trend reversals are usually predicted during crossovers of the signal line and the MACD line

# In[ ]:


def addMACD(df):
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    signal_line = df['close'].ewm(span=9).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df.macd.ewm(span=9, adjust=False).mean()
    df['macdh'] = df['macd'] - df['macd_signal']
    return df

test_asset = addMACD(test_asset)
fig, ax = plt.subplots()
ax.plot(test_asset.index, test_asset['macd'], c='green')
# ax.bar(, height=)
ax.plot(test_asset.index, test_asset['macd_signal'], c='blue')
ax.axhline(y=0,color='black')
ax.fill_between(test_asset.index, test_asset['macdh'], color = 'gray', alpha=0.5, label='MACD Histogram')
ax.set_xticklabels(test_asset['time'].reindex(ax.get_xticks()))
plt.legend()
p = plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, fontsize=8)

