#!/usr/bin/env python
# coding: utf-8

# ## Timeseries analysis and forecast
# 
# This notebook represents my sandbox environment for playing around with timeseries analysis and forecast. The different sections will check for trend and seasonality and will give a quick forecast into the future.
# We will start to use a real world statistics about the number of concurrent users of a mobile app in a hourly resolution over a period of one week. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt


# ### Load the user count timeseries
# Load the user count timeseries from the data set and convert the time column into datetime datatype.

# In[ ]:


# load user count timeseries
data = pd.read_csv('../input/app.csv', sep=';')
print(data.head())
# convert time column to datetime data type
data['time'] = pd.to_datetime(data['time'])
print(data.dtypes)
# use time column as index within the dataset
data = data.set_index('time')
print(data.index)


# ### Plot the user count timeseries
# Simply plot the original timeseries of user counts.

# In[ ]:


print(data.columns.tolist())
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.legend(loc='best')


# ### Lets check for stationary
# Within timeseries analysis, stationary means that the properties of the timeseries such as **mean and variance remain constant over time**. In case mean and variance are increasing or decreasing over time we speak of a trend. 
# Lets do some test for checking a timeseries' stationarity characteristics. A simple but rather visual approach is to plot the moving average as well as the moving variance over time.
# 

# In[ ]:


moving_mean = data['users'].rolling(5).mean()
moving_std = data['users'].rolling(5).std()
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(moving_mean, color='red', label='Moving mean')
plt.plot(moving_std, color='black', label = 'Moving std')
plt.legend(loc='best')


# ### Dickey-Fuller test for stationarity
# 
# Another test for stationarity is given by the Dickey-Fuller test. The Dickey-Fuller test defines a Null Hypothesis (H0) that the timeseries is time dependent and not stationary. If Null Hypothesis can be rejected this means that the timeseries is stationary. 
# The result of the test is interpreted through the tests p-value. A p-value below 5% or 1% means that we can reject the null hypothesis (stationary), otherwise a p-value above means we fail to reject the null hypothesis (non-stationary). 
# * p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
# * p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
# 
# Our example timeseries has a p-value of 0.00027% that allows us to reject the Null Hypothesis, it is clearly stationary and shows no trend. 
# The probability of -3.47% at the significance level of 1% suggests that we can reject the Null Hypothesis with a significance level of less than 1%, meaning a very low probability that we wrongly rejected the hypothesis.

# In[ ]:


from statsmodels.tsa.stattools import adfuller

dftest = adfuller(data['users'].values)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)



# ## Transform a timeseries from non-stationary to stationary [1]
# 
# As mentioned above, there are two aspects that renders a timeseries non-stationary, which are:
# 1. Trend: Mean increases or decreases over time, e.g.: number of users slightly increases or decreases over time
# 2. Seasonality: Mean changes with a specific frequency, e.g.: high mean top of the day, lower mean after business hours.
# 
# The idea is to detect trends and seasonality of a timeseries, use this information about trent and seasonality to transform the timeseries into a stationary timeseries. Then forcast the stationary timeseries and again apply the trend as well as seasonal changes for getting the final prediction.

# ### Difference 
# Subtract the original ts by its shifted value over time. 

# In[ ]:


ts_diff = data['users'] - data['users'].shift()
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(ts_diff, color='red', label='Ts diff')
plt.legend(loc='best')


# ### Decomposing
# A decomposing step models the trend, seasonality as well as residuals individually. 

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data['users'])

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(trend, color='red', label='Trend')
plt.plot(seasonal, color='black', label='Seasonality')
plt.plot(residual, color='green', label='Residuals')
plt.legend(loc='best')


# ## Forecasting a timeseries
# Now that we have extracted the essential residual timeseries by removing seasonality and trend aspects, we can start to predict the timeseries into the future.
# 
# As each timeseries contains multiple parts such as: 
# - Trend 
# - Seasonality 
# - Residuals 
# 
# 
# ### Prediction with Autoregression (AR)
# The autoregression (AR) method models the next step in the sequence as a linear function of the observations at prior time steps [2]. 
# 

# In[ ]:


from statsmodels.tsa.ar_model import AR

# fit model
model = AR(data['users'])
model_fit = model.fit()
# make prediction
pred_users = model_fit.predict(100, 300)
# plot
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(pred_users, color='red', label='Prediction')
plt.legend(loc='best')


# ### Moving Average (MA)
# 
# This forcast method models the next steps within the predicted timeseries through a mean process of historic measurements.
# 
# 

# In[ ]:


from statsmodels.tsa.arima_model import ARMA
# fit model
model = ARMA(data['users'], order=(0, 1))
model_fit = model.fit(disp=False)
# make prediction
ma_pred_users = model_fit.predict(100, 300)

plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(ma_pred_users, color='red',label='User count prediction')
plt.legend(loc='best')


# ### SARIMA
# 

# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(data['users'])
model_fit = model.fit()
# make prediction
sarima_pred = model_fit.predict(100, 300)
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(sarima_pred, color='red',label='User count prediction')
plt.legend(loc='best')


# ### Holt Winters Simple Exponential Smoothing

# In[ ]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# fit model
model = SimpleExpSmoothing(data['users'])
model_fit = model.fit()
# make prediction
pred_holtwint = model_fit.predict(100, 300)
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(pred_holtwint, color='red',label='User count prediction')
plt.legend(loc='best')


# ### Holt Winters Triple Exponential Smoothing
# 

# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
# fit model
model = ExponentialSmoothing(data['users'])
model_fit = model.fit()
# make prediction
hwt_pred = model_fit.predict(100, 300)
plt.figure(figsize=(20, 3))
plt.plot(data['users'], color='blue',label='User count')
plt.plot(hwt_pred, color='red',label='User count prediction')
plt.legend(loc='best')


# ## References
# Many thanks to following important references, which are:
# 
# 1. https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
# 2. https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/
