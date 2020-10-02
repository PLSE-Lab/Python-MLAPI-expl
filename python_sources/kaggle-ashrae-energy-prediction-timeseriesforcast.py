#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
from pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.ticker as ticker


# In[ ]:


df_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv', parse_dates=['timestamp'])
df_weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv',parse_dates=['timestamp'])
df_meta_data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')


# Data Analysis.

# In[ ]:


df_meta_data.info()


# In[ ]:


df_weather_train.info()


# In[ ]:


df_train.info()


# Correlation Check

# In[ ]:


df_train.corrwith(df_weather_train.air_temperature)


# In[ ]:


df_train.corrwith(df_weather_train.cloud_coverage)


# In[ ]:


df_train.corrwith(df_weather_train.wind_direction)


# In[ ]:


df_train.corrwith(df_weather_train.wind_speed)


# In[ ]:


df_train.corrwith(df_weather_train.sea_level_pressure)


# Correlation looks low between training data and weather data. So continue with Training data only .

# Merging of Training data with MetaData to include site_id variable

# In[ ]:


df_merge_train_md = pd.merge(df_train,df_meta_data, how='left', on='building_id')


# In[ ]:


df_merge_train_md.count()


# Converting meter reading of site_id "0" to kwH

# In[ ]:


df_merge_train_md.loc[(df_merge_train_md['site_id'] == 0) & (df_merge_train_md['meter'] == 0),'meter_reading'] =       df_merge_train_md[(df_merge_train_md['site_id'] == 0) & (df_merge_train_md['meter'] == 0)]       ['meter_reading'] * 0.293


# Filtering data based on meter type usage like 0,1,2 & 3

# In[ ]:


df_meterType_0_train = df_merge_train_md[(df_merge_train_md.meter == 0) & (df_merge_train_md.meter_reading>0)]
df_meterType_1_train = df_merge_train_md[(df_merge_train_md.meter == 1) & (df_merge_train_md.meter_reading>0)]
df_meterType_2_train = df_merge_train_md[(df_merge_train_md.meter == 2) & (df_merge_train_md.meter_reading>0)]
df_meterType_3_train = df_merge_train_md[(df_merge_train_md.meter == 3) & (df_merge_train_md.meter_reading>0)]


# In[ ]:


df_meterType_3_train.head()


# As there is no relation among data, so either we can use non parametric model or time series. Considering time series for now to forecast the meter reading

# **TimeSeries Data Exploration**

# In[ ]:


df_mt_0_train_ts = df_meterType_0_train[['timestamp','meter_reading']]
df_mt_1_train_ts = df_meterType_1_train[['timestamp','meter_reading']]
df_mt_2_train_ts = df_meterType_2_train[['timestamp','meter_reading']]
df_mt_3_train_ts = df_meterType_3_train[['timestamp','meter_reading']]


# In[ ]:


df_mt_0_train_ts.set_index('timestamp', inplace=True)
df_mt_1_train_ts.set_index('timestamp', inplace=True)
df_mt_2_train_ts.set_index('timestamp', inplace=True)
df_mt_3_train_ts.set_index('timestamp', inplace=True)


# In[ ]:


df_mt_0_train_ts.plot()


# In[ ]:


df_mt_1_train_ts.plot()


# In[ ]:


df_mt_2_train_ts.plot()


# In[ ]:


df_mt_3_train_ts.plot()


# **Converting daily to weekly meter reading for each meter type**

# In[ ]:


df_mt_0_train_ts_w=df_mt_0_train_ts.meter_reading.resample('w').sum()
df_mt_1_train_ts_w=df_mt_1_train_ts.meter_reading.resample('w').sum()
df_mt_2_train_ts_w=df_mt_2_train_ts.meter_reading.resample('w').sum()
df_mt_3_train_ts_w=df_mt_3_train_ts.meter_reading.resample('w').sum()


# In[ ]:


df_mt_0_train_ts_w.head()


# In[ ]:


df_mt_1_train_ts_w.head()


# In[ ]:


df_mt_2_train_ts_w.head()


# In[ ]:


df_mt_3_train_ts_w.head()


# In[ ]:


fig, ax = plt.subplots()
ax.plot( df_mt_0_train_ts_w,marker='.', linestyle='-', linewidth=0.5, label='Weekly Meter Type 0',color='green')
ax.plot( df_mt_1_train_ts_w,marker='.', linestyle='-', linewidth=0.5, label='Weekly Meter Type 1',color='red')
ax.plot( df_mt_2_train_ts_w,marker='.', linestyle='-', linewidth=0.5, label='Weekly Meter Type 2',color='blue')
ax.plot( df_mt_3_train_ts_w,marker='.', linestyle='-', linewidth=0.5, label='Weekly Meter Type 3',color='magenta')
plt.legend()


# In[ ]:


fig, ax2 = plt.subplots()
ax2.plot(df_mt_0_train_ts_w, color='black', label='meter type 0')
df_mt_1_train_ts_w.plot(label='meter type 1',color='blue')
df_mt_2_train_ts_w.plot(label='meter type 2',color='red')
df_mt_3_train_ts_w.plot(label='meter type 3',color='green')
ax2.legend()
ax2.set_ylabel('Weekly Total (KWh)');


# **Time Series Decomposition to review trend , seasonal , Residual etc**

# Meter Type 0

# In[ ]:


rcParams['figure.figsize'] = 20, 8
decomposition = sm.tsa.seasonal_decompose(df_mt_0_train_ts_w, model='additive')
fig = decomposition.plot()


# Meter Type -1

# In[ ]:


rcParams['figure.figsize'] = 20, 8
decomposition1 = sm.tsa.seasonal_decompose(df_mt_1_train_ts_w, model='multiplicative')
fig1 = decomposition1.plot()


# Meter Type -2

# In[ ]:


rcParams['figure.figsize'] = 20, 8
decomposition2 = sm.tsa.seasonal_decompose(df_mt_2_train_ts_w, model='additive')
fig2 = decomposition2.plot()


# Meter Type -3 

# In[ ]:


rcParams['figure.figsize'] = 20, 8
decomposition3 = sm.tsa.seasonal_decompose(df_mt_3_train_ts_w, model='additive')
fig3 = decomposition3.plot()


# **Testing for Stationarity**

# **Dickey-Fuller Test Function**

# In[ ]:


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(7).mean()
    rolstd = timeseries.rolling(7).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# **Meter Type 0 Stationary Test**

# In[ ]:


test_stationarity(df_mt_0_train_ts_w)


# **Meter Type 1 Stationary Test**

# In[ ]:


test_stationarity(df_mt_1_train_ts_w)


# **Meter Type 2 Stationary Test**

# In[ ]:


test_stationarity(df_mt_2_train_ts_w)


# **Meter Type 3 Stationary Test**

# In[ ]:


test_stationarity(df_mt_3_train_ts_w)


# **Transformation and Differencing**

# **Meter Type 0 Transformation **

# In[ ]:


ts_log_0 = np.log(df_mt_0_train_ts_w)
ts_log_diff_0 = ts_log_0 - df_mt_0_train_ts_w.shift()
plt.plot(ts_log_diff_0)


# **Meter Type 1 Transformation**

# In[ ]:


ts_log_1 = np.log(df_mt_1_train_ts_w)
ts_log_diff_1 = ts_log_1 - df_mt_1_train_ts_w.shift()
plt.plot(ts_log_diff_1)


# **Meter Type 2 Transformation**

# In[ ]:


ts_log_2 = np.log(df_mt_2_train_ts_w)
ts_log_diff_2 = ts_log_2 - df_mt_2_train_ts_w.shift()
plt.plot(ts_log_diff_2)


# **Meter Type 3 Transformation**

# In[ ]:


ts_log_3 = np.log(df_mt_3_train_ts_w)
ts_log_diff_3 = ts_log_3 - df_mt_3_train_ts_w.shift()
plt.plot(ts_log_diff_3)


# **ACF and PACF plots:**
# 

# **Meter Type - 0**

# In[ ]:


plot_acf(df_mt_0_train_ts_w,title='Test',lags=50);


# In[ ]:


lag_acf_0 = acf(df_mt_0_train_ts_w, nlags=20)
lag_pacf_0 = pacf(df_mt_0_train_ts_w, nlags=20, method='ols')


# **ACF**

# In[ ]:


#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf_0)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_mt_0_train_ts_w)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_mt_0_train_ts_w)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


# **PACF**

# In[ ]:


#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf_0)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_mt_0_train_ts_w)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_mt_0_train_ts_w)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[ ]:


plot_pacf(df_mt_0_train_ts_w,title='PACF',lags=20);


# **Meter Type 1: ACF and PACF**

# In[ ]:


plot_acf(df_mt_1_train_ts_w,title='Test',lags=50);


# In[ ]:


lag_acf_1 = acf(df_mt_1_train_ts_w, nlags=20)
lag_pacf_1 = pacf(df_mt_1_train_ts_w, nlags=20, method='ols')


# In[ ]:


plt.subplot(121) 
plt.plot(lag_acf_1)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_mt_1_train_ts_w)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_mt_1_train_ts_w)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plot_acf(df_mt_1_train_ts_w,title='Test',lags=50);


# In[ ]:


#PACF
plt.subplot(122)
plt.plot(lag_pacf_1)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_mt_1_train_ts_w)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_mt_1_train_ts_w)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plot_pacf(df_mt_1_train_ts_w,title='PACF',lags=20);


# **Meter Type 2: ACF and PACF**

# In[ ]:


lag_acf_2 = acf(df_mt_2_train_ts_w, nlags=20)
lag_pacf_2 = pacf(df_mt_2_train_ts_w, nlags=20, method='ols')


# **ACF**

# In[ ]:


plt.subplot(121) 
plt.plot(lag_acf_2)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_mt_2_train_ts_w)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_mt_2_train_ts_w)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plot_acf(df_mt_2_train_ts_w,title='Test',lags=50);


# **PACF**

# In[ ]:


#PACF
plt.subplot(122)
plt.plot(lag_pacf_2)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_mt_2_train_ts_w)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_mt_2_train_ts_w)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plot_pacf(df_mt_2_train_ts_w,title='PACF',lags=20);


# **Meter Type 3: ACF and PACF**

# In[ ]:


lag_acf_3 = acf(df_mt_3_train_ts_w, nlags=20)
lag_pacf_3 = pacf(df_mt_3_train_ts_w, nlags=20, method='ols')


# **ACF**

# In[ ]:


plt.subplot(121) 
plt.plot(lag_acf_3)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_mt_3_train_ts_w)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_mt_3_train_ts_w)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plot_acf(df_mt_3_train_ts_w,title='Test',lags=50);


# **PACF**

# In[ ]:


#PACF
plt.subplot(122)
plt.plot(lag_pacf_3)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_mt_3_train_ts_w)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_mt_3_train_ts_w)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plot_pacf(df_mt_3_train_ts_w,title='PACF',lags=20);


# **AREMA Model**

# **Meter Type 0: AREMA Forecast**

# In[ ]:


model_0 = ARIMA(df_mt_0_train_ts_w, order=(2, 0, 2))  
results_AR_0 = model_0.fit(disp=-1)  
plt.plot(ts_log_diff_0)
plt.plot(results_AR_0.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR_0.fittedvalues-ts_log_diff_1)**2))
results_AR_0.summary()


# In[ ]:


fcast_0 = results_AR_0.predict(len(df_mt_0_train_ts_w),len(df_mt_0_train_ts_w)+10)


# **Forecast**

# In[ ]:


formatter = ticker.StrMethodFormatter('{x:,.0f}')
title = 'Meter Reading of Type - 0'
ylabel='Weekly Meter Reading'
xlabel='' # we don't really need a label here

ax = df_mt_0_train_ts_w.plot(legend=True,figsize=(12,6),title=title)
fcast_0.plot(legend=True,label='Forecast')
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# In[ ]:


print(fcast_0)


# **Meter Type 1: AREMA Forecast**

# In[ ]:


model_1 = ARIMA(df_mt_1_train_ts_w, order=(0, 1, 2))  
results_AR_1 = model_1.fit(disp=-1)  
plt.plot(ts_log_diff_1)
plt.plot(results_AR_1.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR_1.fittedvalues-ts_log_diff_1)**2))
results_AR_1.summary()


# In[ ]:


fcast_1 = results_AR_1.predict(len(df_mt_1_train_ts_w),len(df_mt_1_train_ts_w)+10)


# Forecast 

# In[ ]:


formatter = ticker.StrMethodFormatter('{x:,.0f}')
title = 'Meter Reading of Type - 1'
ylabel='Weekly Meter Reading'
xlabel='' # we don't really need a label here

ax = df_mt_1_train_ts_w.plot(legend=True,figsize=(12,6),title=title)
fcast_1.plot(legend=True,label='Forecast')
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# In[ ]:


print(fcast_1)


# **Meter Type 2: AREMA Forecast**

# In[ ]:


model_2 = ARIMA(df_mt_2_train_ts_w, order=(2, 1, 2))  
results_AR_2 = model_2.fit(disp=-1)  
plt.plot(ts_log_diff_2)
plt.plot(results_AR_2.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR_2.fittedvalues-ts_log_diff_2)**2))
results_AR_2.summary()


# In[ ]:


fcast_2 = results_AR_2.predict(len(df_mt_2_train_ts_w),len(df_mt_2_train_ts_w)+10)


# **Forecast**

# In[ ]:


formatter = ticker.StrMethodFormatter('{x:,.0f}')
title = 'Meter Reading of Type - 1'
ylabel='Weekly Meter Reading'
xlabel='' # we don't really need a label here

ax = df_mt_2_train_ts_w.plot(legend=True,figsize=(12,6),title=title)
fcast_2.plot(legend=True,label='Forecast')
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# In[ ]:


print(fcast_2)


# **Meter Type 3: AREMA Forecast**

# In[ ]:


model_3 = ARIMA(df_mt_3_train_ts_w, order=(2, 0, 2))  
results_AR_3 = model_3.fit(disp=-1)  
plt.plot(ts_log_diff_3)
plt.plot(results_AR_3.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR_3.fittedvalues-ts_log_diff_3)**2))
results_AR_3.summary()


# In[ ]:


fcast_3 = results_AR_3.predict(len(df_mt_3_train_ts_w),len(df_mt_3_train_ts_w)+10)


# **Forecast**

# In[ ]:


formatter = ticker.StrMethodFormatter('{x:,.0f}')
title = 'Meter Reading of Type - 3'
ylabel='Weekly Meter Reading'
xlabel='' # we don't really need a label here

ax = df_mt_3_train_ts_w.plot(legend=True,figsize=(12,6),title=title)
fcast_3.plot(legend=True,label='Forecast')
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel)
ax.yaxis.set_major_formatter(formatter);


# In[ ]:


print(fcast_3)


# In[ ]:





# In[ ]:




