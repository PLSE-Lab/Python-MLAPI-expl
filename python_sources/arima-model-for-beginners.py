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


df = pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv')
df.head()


# converting Month to date type

# In[ ]:


df['Month'] = pd.to_datetime(df['Month'], format = '%Y-%m')


# In[ ]:


df.dtypes


# Set Month as index

# In[ ]:


df = df.set_index('Month')
df.head()


# Plot with Original Data

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:



import matplotlib.pyplot as plt
plt.figure(figsize = (12, 6))
plt.plot(df)
plt.xlabel ('Date')
plt.ylabel ('No. of Passengers')
plt.title('Air Passengers Data', fontsize = 18)


# # Rolling Mean and SD
# 

# In[ ]:


rolmean = df['#Passengers'].rolling(12).mean()
rolsd = df['#Passengers'].rolling(12).std()
print(rolmean, rolsd)


# In[ ]:


plt.figure(figsize = (12, 6))
orign = plt.plot(df, color='blue', label = 'original data')
mean = plt.plot(rolmean, color='red', label = 'Rolling mean data')
std = plt.plot(rolsd, color='black', label = 'Rolling std data')
plt.legend()
plt.xlabel ('Date')
plt.ylabel ('No. of Passengers')
plt.title('Rolling mean & STD data', fontsize = 18)


# # Dickey-Fuller Test

# In[ ]:


# Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller
print('Results of Dickey-Fuller Test: ')
dftest = adfuller(df['#Passengers'], autolag='AIC' )
dftest
dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print(dfoutput)
for key, values in dftest[4].items():
    print(f'critical values ({key}) : {values}')
#     print(f'')
#     dfoutput['critical values(%s)'%key] = values 

# print(dfoutput)


# P-Value and Test Statistic values are high, need to go for log values.
# Making Time Series Stationary

# In[ ]:


df_log = np.log(df)
df_log.head()
plt.figure(figsize = (12, 6))
plt.plot(df_log)


# In[ ]:


rolmean_log = df_log['#Passengers'].rolling(12).mean()
rolstd_log = df_log['#Passengers'].rolling(12).std()
plt.figure(figsize = (12, 6))
plt.plot(df_log, color='blue', label = 'Rolling log Original')
plt.plot(rolmean_log, color='red', label = 'Rolling log mean')
plt.legend()


# In[ ]:


dif_log = df_log['#Passengers'] - rolmean_log
dif_log.head()

# dif_log.rename(column='pas')
dif_log.dropna(inplace = True)
# dif_log
# rolmean_log 
dif_log.head()


# Creating function to re-use for dickey-Fuller test and plots

# In[ ]:


def test_stationary(timeseries):
    # rolling stats
    rolmean1 = timeseries.rolling(12).mean()
    rolstd1 = timeseries.rolling(12).std()
    # Rolling stats plot
    plt.figure(figsize = (12, 6))
    plt.plot(timeseries, color='blue', label = 'Rolling log Original')
    plt.plot(rolmean1, color='red', label = 'Rolling log mean')
    plt.plot(rolstd1, color='black', label = 'Rolling log std')
    plt.legend()
    #Dickey-Fuller Test
    print('Results of Dickey-Fuller Test: ')
    dftest = adfuller(timeseries, autolag='AIC' )
    dftest
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    dfoutput
    for key, values in dftest[4].items():
        dfoutput['critical values(%s)'%key] = values 
    print(dfoutput)
    


# In[ ]:


test_stationary(dif_log)


# In[ ]:


dif_log_sft = df_log['#Passengers'] - df_log['#Passengers'].shift(1)
dif_log_sft.head()
plt.figure(figsize = (12, 6))
plt.plot(dif_log_sft)


# deleting NaN values to avoid issues

# In[ ]:


dif_log_sft.dropna(inplace = True)
dif_log_sft.head()
test_stationary(dif_log_sft)


# Decomposition:
# In this approach, both trend and seasonality are modeled separately and the remaining part of the series is returned.

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(df_log)
trend = decompose.trend
seasonal = decompose.seasonal
residual = decompose.resid

plt.figure(figsize = (12, 6))
plt.subplot(411)
plt.plot(df_log, label = 'Origional')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label = 'Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label = 'Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label = 'Residual')
plt.legend(loc='best')


# In[ ]:


df_log_decompose = residual['#Passengers']
df_log_decompose.dropna(inplace = True) 
df_log_decompose.head(10)
test_stationary(df_log_decompose)


# AR Model

# In[ ]:


from statsmodels.tsa.ar_model import AR
model = AR(dif_log_sft)
model_ar = model.fit()


# In[ ]:


plt.figure(figsize = (12, 6))
plt.plot(dif_log_sft)
plt.plot(model_ar.fittedvalues, color='red' )
plt.title('RSS : %.4f'% np.nansum((model_ar.fittedvalues - dif_log_sft)**2))


# # Reversing the transformations
# # Fitted or predicted values:

# In[ ]:


predict_ARIMA_dif = pd.Series(model_ar.fittedvalues, copy = True)
predict_ARIMA_dif


# In[ ]:


# Cumulative Sum to reverse differencing:
predict_ARIMA_dif_cumsum = predict_ARIMA_dif.cumsum() 
predict_ARIMA_dif_cumsum 


# In[ ]:


# Adding 1st month value which was previously removed while differencing:
predict_ARIMA_log = pd.Series(df_log['#Passengers'].iloc[0], index = df_log.index)
predict_ARIMA_log = predict_ARIMA_log.add(predict_ARIMA_dif_cumsum, fill_value=0)
predict_ARIMA_log.head()


# In[ ]:


# Taking Exponent to reverse Log Transform:
predict_ARIMA = np.exp(predict_ARIMA_log)
predict_ARIMA.head()


# In[ ]:


plt.figure(figsize = (12, 6))
plt.plot(df)
plt.plot(predict_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(np.nansum((predict_ARIMA-df['#Passengers'])**2)/len(df['#Passengers'])))


# #ACF and PACF for p and q values. (before ARIMA)

# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.figure(figsize = (12, 6))
plot_acf(dif_log_sft, ax=plt.gca(), lags = 20) # q 
plt.show()
plt.figure(figsize = (12, 6))
plot_pacf(dif_log_sft, ax=plt.gca(), lags = 20) # p 
plt.show()


# # ARIMA MODEL

# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=1)  


# Try order values ( ex: order=(2, 1, 2) or (1, 1, 1) ) changes until we get RSS less value.

# In[ ]:



plt.figure(figsize = (12, 6))
plt.plot(dif_log_sft)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-dif_log_sft)**2))


# # Reversing the transformations
# # Fitted or predicted values:
# 

# In[ ]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()


# In[ ]:


# Cumulative Sum to reverse differencing:
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()


# In[ ]:


# Adding 1st month value which was previously removed while differencing:
predictions_ARIMA_log = pd.Series(df_log['#Passengers'].iloc[0], index=df_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# In[ ]:


# Taking Exponent to reverse Log Transform:
predict_ARIMA = np.exp(predictions_ARIMA_log)
predict_ARIMA.head()


# In[ ]:


plt.plot(df)
plt.plot(predict_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predict_ARIMA-df['#Passengers'])**2)/len(df['#Passengers'])))


# Predictions: we have 144 rows data and predicting another 10 years (10*12 = 120). total 144+120 = 264.
# pridiction plots:
# 

# In[ ]:


results_ARIMA.plot_predict(1, 264)


# if we want only values:

# In[ ]:


x = results_ARIMA.forecast(steps = 120)
np.exp(x[0])


# Another example ARIMA Model with Train and Test data on this.

# In[ ]:


#divide into train and test set
train = df[:int(0.75*(len(df)))]
test = df[int(0.75*(len(df))):]

plt.plot(train)
plt.plot(test, color='red')


# In[ ]:


from sklearn.metrics import mean_squared_error

model = ARIMA(train, order = (1,1,1))
model_fit = model.fit(disp = 1)


# In[ ]:


start = test.index.min()
end = test.index.max()

predict = model_fit.predict(start = start, end = end)


# In[ ]:


mse = mean_squared_error(df[start : end], predict)
rmse = mse ** 0.5 #(sqrt of mse)
print (f'MSE : {mse} , RMSE : {rmse}')


# In[ ]:


predict_ARIMA_dif = pd.Series(predict, copy = True)
predict_ARIMA_dif.head() 


# In[ ]:


predict_ARIMA_dif_cumsum = predict_ARIMA_dif.cumsum()
predict_ARIMA_dif_cumsum.head() 


# In[ ]:


predict_ARIMA_log = pd.Series(test['#Passengers'].iloc[0], index = test.index)
predict_ARIMA_log = predict_ARIMA_log.add(predict_ARIMA_dif_cumsum, fill_value=0)
predict_ARIMA_log.head()


# In[ ]:


df_sft = df['#Passengers'] - df['#Passengers'].shift()


# In[ ]:


plt.plot(df)
plt.plot(predict_ARIMA_log, color='red')
plt.title('RMSE: %.4f'% np.sqrt(np.nansum((predict_ARIMA_log-df_sft)**2)/len(df_sft)))


# Fell free to comment with suggestions or any clarifications for further active discussions. 
# My e-mailid is mcommahesh@gmail.com
# 
