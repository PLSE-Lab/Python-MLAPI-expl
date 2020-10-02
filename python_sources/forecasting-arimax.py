#!/usr/bin/env python
# coding: utf-8

#  **1. Exploring and analyse the time series data.
#  2. Forecasting using ARIMA, ARIMAX**
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dateparse = lambda d : dt.datetime.strptime(d, '%Y-%m')
data = pd.read_csv("../input/AirPassengers.csv", index_col='Month',date_parser=dateparse)
data.info()


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


#x = [dt.datetime.strptime(d,'%Y-%m') for d in data.index]
#data.set_index = [ str(d.year)+"-"+str(d.month)+"-"+str(d.day) for d in x]
data.index


# In[ ]:


data['#Passengers']['1949-01-01']


# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(data['#Passengers'])


# In[ ]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    plt.figure(figsize=(12,8))
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

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
    print(dfoutput)


# In[ ]:


test_stationarity(data['#Passengers'])


# In[ ]:


ts_log = np.log(data['#Passengers'])
plt.figure(figsize=(12,8))
plt.plot(ts_log)


# In[ ]:


test_stationarity(np.log(data['#Passengers']))


# In[ ]:


moving_avg = np.log(data['#Passengers']).rolling(12).mean()
plt.plot(np.log(data['#Passengers']))
plt.plot(moving_avg, color='red')


# In[ ]:


ts_log_moving_avg_diff = np.log(data['#Passengers']) - moving_avg
ts_log_moving_avg_diff.head(12)


# In[ ]:


ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)


# Using **Exponential Weighted Average**. Recent values given more weight.

# In[ ]:


expwighted_avg = np.log(data['#Passengers']).ewm(halflife=12).mean()
plt.plot(np.log(data['#Passengers']))
plt.plot(expwighted_avg, color='red')


# In[ ]:


ts_log_ewma_diff = np.log(data['#Passengers']) - expwighted_avg
test_stationarity(ts_log_ewma_diff)


# In[ ]:


moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')


# In[ ]:


ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)


# In[ ]:


expwighted_avg = ts_log.ewm(halflife=12).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')


# In[ ]:


ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)


# In[ ]:


ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)


# In[ ]:


ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# fit model
model = ARIMA(ts_log, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[ ]:


# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()


# In[ ]:


residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# In[ ]:


validation_size =int(len(ts_log)*0.66)
X = data['#Passengers'].astype('float')
train, test = X[0:validation_size], X[validation_size:len(X)]
history = [x for x in train]
predictions = list()


# In[ ]:


for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs,y = test[t],test.index[t]
	history.append(obs)
	print('predicted=%f, expected=%f, month=%s' % (yhat, obs, y))


# In[ ]:


error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# In[ ]:


plt.figure(figsize=(12,8))

print(type(predictions),type(history))
plt.plot(predictions, color='red',label="Predictions")
plt.plot(history,label="Label")

