#!/usr/bin/env python
# coding: utf-8

# 

# https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c
# 
# https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
# 
# https://towardsdatascience.com/forecasting-exchange-rates-using-arima-in-python-f032f313fc56

# In[ ]:


# https://machinelearningmastery.com/make-sample-forecasts-arima-python/


# In[1]:


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


# In[3]:


# line plot of time series
from pandas import Series
from matplotlib import pyplot
# load dataset
series = Series.from_csv('../input/daily-minimum-temperatures.csv', header=0)
# display first few rows
print(series.head(20))
# line plot of dataset
series.plot()
pyplot.show()


# In[4]:


# split the dataset
split_point = len(series) - 7
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))


# In[15]:


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


# In[16]:


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# In[71]:


differenced[3260:3280]


# In[72]:


from statsmodels.tsa.arima_model import ARIMA

# load dataset
series = dataset
# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(12,1,0))
model_fit = model.fit(disp=0)
# print summary of fit model
print(model_fit.summary())


# In[59]:


# one-step out-of sample forecast
forecast = model_fit.forecast(steps=7)[0]


# In[56]:


X[len(X) -365]


# In[60]:


# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)
print('Forecast: %f' % forecast)


# In[28]:


# one-step out of sample forecast
start_index = len(differenced)
end_index = len(differenced)
forecast = model_fit.predict(start=start_index, end=end_index)


# In[29]:


# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)
print('Forecast: %f' % forecast)


# In[73]:


model_fit.forecast(steps=7)


# In[62]:


# multi-step out-of-sample forecast
forecast = model_fit.forecast(steps=7)[0]
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1


# In[31]:


start_index = len(differenced)
end_index = start_index + 6
forecast = model_fit.predict(start=start_index, end=end_index)
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1


# In[ ]:




