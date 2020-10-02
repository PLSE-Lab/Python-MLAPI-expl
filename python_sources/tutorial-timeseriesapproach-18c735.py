#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from kaggle.competitions import twosigmanews

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller


# In[ ]:


env = twosigmanews.make_env()


# In[ ]:


market_data = env.get_training_data()[0]


# In[ ]:


def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window=120).mean()
    rolstd = timeseries.rolling(window=120).std()

    #Plot rolling statistics:
    plt.figure(figsize=(20,10))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[ ]:


time_series_df = market_data[["time"]].groupby(by=["time"]).size()

market_data = market_data[['assetCode', 'time','close']]
grouped = market_data.groupby('assetCode').groups
# print(grouped.keys())

# asset = 'AAPL.O'
asset = 'APH.N'

asset_data = market_data[(market_data.assetCode == asset)]

asset_series = pd.Series(asset_data.close.values, index=asset_data.time)
asset_series = asset_series['2016-05':]

ts_log = np.log(asset_series)


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log,freq=10)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig,ax = plt.subplots(figsize=(15,20))
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[ ]:


ts_log_diff = residual
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)


# In[ ]:


#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

model = ARIMA(ts_log, order=(7, 1, 6))
results_ARIMA = model.fit(disp=-1)  
fig,axes = plt.subplots(1,1,figsize=(20,10))
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))


# In[ ]:





# In[ ]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
pd.concat([ts_log.tail(20), predictions_ARIMA_log.tail(20)],axis=1)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(asset_series)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-asset_series)**2)/len(asset_series)))


# In[ ]:


# It doesn't work yet
# def make_my_predictions(market_obs_df, news_obs_df, predictions_template_df):
#     for (asset in grouped.keys()):
#         asset_data = market_data[(market_data.assetCode == asset)]

#         asset_series = pd.Series(asset_data.close.values, index=asset_data.time)

#         ts_log = np.log(asset_series)
#         ts_log_diff = ts_log - ts_log.shift()
#         ts_log_diff.dropna(inplace=True)
        
#         model = ARIMA(ts_log, order=(2, 1, 0))  
#         results_ARIMA = model.fit(disp=-1)
        
#         predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
#         predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#         predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
#         predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
        
#         predictions_template_df.confidenceValue = 0.5
#         return predictions_template_df

def make_my_predictions(market_obs_df, news_obs_df, predictions_template_df):
    predictions_template_df.confidenceValue = 1.0
    return predictions_template_df

for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
  predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
  env.predict(predictions_df)

env.write_submission_file()

