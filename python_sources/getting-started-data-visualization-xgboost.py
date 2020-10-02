#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# ## University of Liverpool Ion Switching

# This is my first data analysis notebook and pleaes **UPVOTE** for this notebook if you find it useful!

# The focus of this notebook is for data visualization and it will focus on using SNS as the plotting library. It hopes to use the plots to help each other to find possible better features.

# # Import Configuration

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn import *
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import scipy as sp
sns.set(style="whitegrid")


# # Configuration

# In[ ]:


PLOT_ALL = True


# # Load Data

# In[ ]:


df_train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
df_test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')


# In[ ]:


df = pd.concat([df_train.drop(['open_channels'], axis=1), df_test])
df_train.shape, df_test.shape, df.shape


# # Configuration

# In[ ]:


batch_time = 500000 # The signal batch time period


# # Data Overview

# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


df.describe()


# # Channel Distribution Visualization

# In[ ]:


oc_vc = df_train['open_channels'].value_counts()
ax = sns.barplot(x=oc_vc.index, y=oc_vc.values)


# In[ ]:


if PLOT_ALL:
    plt.figure(figsize=(12,12))
    for i in range(10):
        plt.subplot(5,2,i+1)
        idx = range(batch_time*i, batch_time*(i+1)-1)
        oc_vc = df_train.loc[idx, 'open_channels'].value_counts()
        ax = sns.barplot(x=oc_vc.index, y=oc_vc.values)


# The plot above is quite interesting, showing that in some batches, there are only 0 or 1 open channels, where in other batches, there are open channels variation from 1-5.

# In[ ]:


if PLOT_ALL:
    plt.figure(figsize=(12,12))
    for i in range(10):
        plt.subplot(5,2,i+1)
        idx = range(batch_time*i, batch_time*(i+1)-1)
        ax = sns.lineplot(x="time", y="open_channels", data=df_train[batch_time*i:batch_time*(i+1)-1])


# # Signal Distribution Visualization

# ## Signal distribution for different open channel numbers

# In[ ]:


idx = 0
plt.figure(figsize=(18,24))
for i, d in df_train.groupby('open_channels'):
    plt.subplot(4,3,idx+1)
    sns.distplot(d['signal'], bins=50)
    plt.title('Signal Distribution for %d number of open channels' % i)
    idx += 1


# In[ ]:


if PLOT_ALL:
    sns.pairplot(x_vars=["signal"], y_vars=["open_channels"], data=df_train, size=6)
    plt.title('Signal Distribution for different number of open channels')


# ## Signal in Time vs Open Channel

# In[ ]:


non_zero_chn_index = df_train.index[df_train['open_channels'] > 0]
print(non_zero_chn_index.shape)

def plot_signal(mid_idx, plot_len):
    plt.figure(figsize=(12,12))
    for i in range(1,3):
        start = mid_idx[i-1] - plot_len
        end = mid_idx[i-1] + plot_len

        plt.subplot(2,2,i)
        plt.title('Open Channels, Time: %d - %d' % (start, end))
        sns.lineplot(df_train.loc[start:end, 'time'], df_train.loc[start:end, 'open_channels'])

        plt.subplot(2,2,i+2)
        plt.title('Signal, Time: %d - %d' % (start, end))
        sns.lineplot(df_train.loc[start:end, 'time'], df_train.loc[start:end, 'signal'])


# In[ ]:


if PLOT_ALL:
    mid_idx = [non_zero_chn_index[100000], non_zero_chn_index[1759848]]
    plot_len = 1000
    plot_signal(mid_idx, plot_len)


# From this plot, it does look like that higher signal strength leads to a larger number of open channels.

# In[ ]:


if PLOT_ALL:
    mid_idx = [non_zero_chn_index[10000], non_zero_chn_index[1759848]]
    plot_len = 20
    plot_signal(mid_idx, plot_len)


# From the zoomed in plots, it does look like the shape of the signal and the shape of the number of open channels have a strong correlation. The number of open channels look like a delayed version of the input signal.

# In[ ]:


if PLOT_ALL:
    plt.figure(figsize=(6,6))
    sns.distplot(df_train['signal'], bins=20)
    sns.distplot(df_test['signal'], bins=20)
    plt.title('Signal Distribution for Test and Train')
    plt.legend(labels=['Train', 'Test'])


# # Lag plot of signals

# In[ ]:


plt.figure(figsize=(18,24))
for i in range(12):
    ax = plt.subplot(4,3,i+1)
    batch_idx = 2
    pd.plotting.lag_plot(df_train['open_channels'][batch_time*batch_idx:batch_time*(batch_idx+1)-1], lag=i+1, ax=ax)
    plt.title('Signal lag plot for batch %d' % (i+1))


# ## Autocorrelation of signals
# Autocorrelation is the correlation of a signal with a delayed copy of itself as a function of delay. This will show whether there are repetitive patterns inside different batches of signal.

# In[ ]:


plt.figure(figsize=(18,24))
for i in range(10):
    ax = plt.subplot(4,3,i+1)
    pd.plotting.autocorrelation_plot(df['signal'][batch_time*i:batch_time*(i+1)-1], ax=ax)
    plt.ylim([-0.25, 0.25])
    plt.title('Signal Autocorrelation for batch %d' % (i+1))


# For batch 1-4, there seems to be a more repetitive pattern based on the autocorrelation plot.

# ## Frequency Domain

# In[ ]:


f_s = 10000

plt.figure(figsize=(18,18))
for i in range(10):
    plt.subplot(5,2,i+1)
    x = df['signal'][batch_time*i:batch_time*(i+1)-1]
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x)) * f_s

    start = 0
    end = len(x) // 100
    plt.plot(freqs[start:end], np.abs(X)[start:end])
    plt.title("Frequency spectrum of the signal, Batch %d" % (i+1))


# This shows that all the active signals are below 100 Hz, except batch 4 has a particularly noisy signal.

# # Feature Engineering

# In[ ]:


def add_window_feature(df):
    window_sizes = [10, 25, 50, 100, 500, 1000, 5000, 10000, 25000]
    for window in window_sizes:
        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()
        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()
        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()
        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()
        
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    
    return df


# In[ ]:


df_train = add_window_feature(df_train)
print(df_train.columns)
df_test = add_window_feature(df_test)
print(df_test.columns)


# # Modeling

# In[ ]:


X = df_train.drop(['time', 'open_channels'], axis=1)
y = df_train['open_channels']


# ## XGBoost

# In[ ]:


model = xgb.XGBRegressor(max_depth=3)
model.fit(X,y)


# ## LGBM

# In[ ]:


model = lgbm.LGBMRegressor(n_estimators=100)
model.fit(X, y)


# # Prediction

# In[ ]:


X_test = df_test.drop(['time'], axis=1)
preds = model.predict(X_test)


# In[ ]:


df_test['open_channels'] = np.round(np.clip(preds, 0, 10)).astype(int)
df_test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')


# This is the end of my first EDA notebook. Any feedback or comments are greatly welcomed!
