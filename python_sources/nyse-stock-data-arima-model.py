#!/usr/bin/env python
# coding: utf-8

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


# # ARIMA model for NYSE stock data

# ### Import other modules

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA


# ### Import data and check head

# In[ ]:


df = pd.read_csv('../input/prices-split-adjusted.csv', index_col=0)
df.head()


# ### Input from user to select company for ARIMA model

# In[ ]:


# Filter dataframe only for chosen ticker symbol
dfa = df[df['symbol'] == 'AAPL']


# In[ ]:


dfa.head()


# In[ ]:


dfa.index.sort_values()


# In[ ]:


# Convert index to pandas datetime
dfa.index = pd.to_datetime(dfa.index, format="%Y/%m/%d")


# In[ ]:


df_final = dfa.drop(['symbol','open','low','high','volume'], axis=1)


# In[ ]:


# Conver to Series to run Dickey-Fuller test
df_final = pd.Series(df_final['close'])


# In[ ]:


type(df_final)


# ## Function to check stationarity

# In[ ]:


def check_stationarity(ts_data):
    
    # Rolling statistics
    roll_mean = ts_data.rolling(30).mean()
    roll_std = ts_data.rolling(5).std()
    
    # Plot rolling statistics
    fig = plt.figure(figsize=(20,10))
    plt.subplot(211)
    plt.plot(ts_data, color='black', label='Original Data')
    plt.plot(roll_mean, color='red', label='Rolling Mean(30 days)')
    plt.legend()
    plt.subplot(212)
    plt.plot(roll_std, color='green', label='Rolling Std Dev(5 days)')
    plt.legend()
    
    # Dickey-Fuller test
    print('Dickey-Fuller test results\n')
    df_test = adfuller(ts_data, regresults=False)
    test_result = pd.Series(df_test[0:4], index=['Test Statistic','p-value','# of lags','# of obs'])
    print(test_result)
    for k,v in df_test[4].items():
        print('Critical value at %s: %1.5f' %(k,v))
    


# In[ ]:


check_stationarity(df_final)


# #### As Test statistic is greater than all critical values, the time series is clearly not stationary. Testing different transformations for stationarity is required before applying ARIMA model to time series.

# ### Log transformation of original time series

# In[ ]:


# Log transform time series
df_final_log = np.log(df_final)
df_final_log.head()


# In[ ]:


# Check stationarity
df_final_log.dropna(inplace=True)
check_stationarity(df_final_log)


# ### The log transformation is not stationary as test statistic is greater than critical values and you can also visualize this on the 1st graph.

# ### Log differencing transformation of original time series

# In[ ]:


# Log Differencing
df_final_log_diff = df_final_log - df_final_log.shift()


# In[ ]:


df_final_log_diff.dropna(inplace=True)
check_stationarity(df_final_log_diff)


# ### Simple differencing transformation of original time series

# In[ ]:


# Differencing
df_final_diff = df_final - df_final.shift()


# In[ ]:


df_final_diff.dropna(inplace=True)
check_stationarity(df_final_diff)


# ### As simple differencing yields a test statistic much lower than critical values, we will use this for applying ARIMA.

# In[ ]:


from statsmodels.tsa.stattools import acf, pacf


# In[ ]:


df_acf = acf(df_final_diff)


# In[ ]:


df_pacf = pacf(df_final_diff)


# In[ ]:


import statsmodels.api as sm


# In[ ]:


fig1 = plt.figure(figsize=(20,10))
ax1 = fig1.add_subplot(211)
fig1 = sm.graphics.tsa.plot_acf(df_acf, ax=ax1)
ax2 = fig1.add_subplot(212)
fig1 = sm.graphics.tsa.plot_pacf(df_pacf, ax=ax2)


# In[ ]:


model = ARIMA(df_final_diff, (1,1,0))


# In[ ]:


fit_model = model.fit(full_output=True)


# In[ ]:


predictions = model.predict(fit_model.params, start=1760, end=1769)


# In[ ]:


fit_model.summary()


# In[ ]:


predictions


# In[ ]:


fit_model.predict(start=1760, end=1769)


# ### Time to re-transform data back to original scale

# In[ ]:


pred_model_diff = pd.Series(fit_model.fittedvalues, copy=True)
pred_model_diff.head()


# In[ ]:


# Calculate cummulative sum of the fitted values (cummulative sum of differences)
pred_model_diff_cumsum = pred_model_diff.cumsum()
pred_model_diff_cumsum.head()


# In[ ]:


# Element-wise addition back to original time series
df_final_trans = df_final.add(pred_model_diff_cumsum, fill_value=0)
# Last 5 rows of fitted values
df_final_trans.tail()


# In[ ]:


# Last 5 rows of original time series
df_final.tail()


# In[ ]:


# Plot of orignal data and fitted values
plt.figure(figsize=(20,10))
plt.plot(df_final, color='black', label='Original data')
plt.plot(df_final_trans, color='red', label='Fitted Values')
plt.legend()


# In[ ]:


x = df_final.values
y = df_final_trans.values


# In[ ]:


# Trend of error
plt.figure(figsize=(20,8))
plt.plot((x - y), color='red', label='Delta')
plt.axhline((x-y).mean(), color='black', label='Delta avg line')
plt.legend()


# ### Average error appears to be around $0.35 per share

# ## Final step is to create a Series with ten prediction values

# In[ ]:


final_pred = []
for i in predictions:
    t = df_final[-1] + i
    final_pred.append(t)


# In[ ]:


final_pred = pd.Series(final_pred)
final_pred


# ### This is my first kernel. As the data ends by 2016, you can compare it to actual share of the comapny chosen by the user to stock data from Google or Yahoo finance. Comments are welcome!

# In[ ]:




