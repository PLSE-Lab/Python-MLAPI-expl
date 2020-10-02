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


# # Importing the below '**warnings**' package in order to ignore some unimportant warnings

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# # Importing dataset with date as Index with DateTimestamp

# In[ ]:


df = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv', index_col='date', parse_dates=True)
df.head()


# # Setting the frequency as Daily 'D'

# In[ ]:


df.index.freq='D'


# # Checking if the dataset have missing values

# In[ ]:


df.info()


# # Visualization of 'meantemp' values for 2013 - 2017

# In[ ]:


df['meantemp'].plot(figsize=(15,5));


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


# # ETS Decomposition

# In[ ]:


seasonal_decompose(df['meantemp']).plot();


# In[ ]:


seasonal_decompose(df['meantemp']).observed.plot(figsize=(12,4));


# In[ ]:


seasonal_decompose(df['meantemp']).resid.plot(figsize=(12,4));


# In[ ]:


seasonal_decompose(df['meantemp']).trend.plot(figsize=(12,4));


# In[ ]:


seasonal_decompose(df['meantemp']).seasonal.plot(figsize=(12,4));


# # Augmented Dickey Fuller Test for Stationarity of Data

# In[ ]:


from statsmodels.tsa.stattools import adfuller


# In[ ]:


def test_adfuller(df):
    result = adfuller(df)
    labels = ['adf value', 'p-value', '# lags', '# observation']
    out = pd.Series(result[:4], index=labels)
    
    for key,val in result[4].items():
        out['critical '+key]=val
    
    print(out)
    
    if(result[1] <= 0.5):
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[ ]:


test_adfuller(df['meantemp'])


# # Run pmdarima.auto_arima to obtain the recommended orders with least AIC

# In[ ]:


# !pip install pmdarima --> run at console
from pmdarima import auto_arima


# In[ ]:


auto_arima(df['meantemp'], seasonal=False).summary()


# # Splitting Train and Test Data

# In[ ]:


len(df)


# In[ ]:


train = df[:-365]
test = df[-365:]


# In[ ]:


len(train)


# In[ ]:


len(test)


# # Fit ARIMA(1, 1, 1) model

# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA, ARMA


# In[ ]:


model = ARIMA(train['meantemp'],order=(1,1,1))
results = model.fit()
results.summary()


# # Prediction for Test Data

# In[ ]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(1,1,1) Predictions')


# # Visualization of Test Data vs Predicted Data

# In[ ]:


test['meantemp'].plot(legend=True,figsize=(12,6))
predictions.plot(legend=True)


# # Resampling to Monthly Dataset as Monthly Climate Change gives more insights

# In[ ]:


resample_monthly_temp = df['meantemp'].resample('MS').mean()


# # Visulization of Monthly Temperature

# In[ ]:


resample_monthly_temp.plot(figsize=(12,6))


# In[ ]:


len(resample_monthly_temp)


# In[ ]:


resample_monthly_temp.head()


# # ETS Decompositon

# In[ ]:


seasonal_decompose(resample_monthly_temp).plot();


# # Performing ADF test for stationarity of Data

# In[ ]:


test_adfuller(resample_monthly_temp)


# # Run pmdarima.auto_arima to obtain the recommended orders with least AIC

# In[ ]:


auto_arima(resample_monthly_temp,seasonal=True, m=12).summary()


# # Spliting train/test data

# In[ ]:


train = resample_monthly_temp[:-13]
test = resample_monthly_temp[-13:]


# In[ ]:


test


# # Fit SARIMAX(2, 0, 2)x(2, 1, 0, 12) model

# In[ ]:


model = SARIMAX(train,order=(2,0,2), seasonal_order=(2,1,[],12), m=12)
results = model.fit()
results.summary()


# # Prediction and Evaluation

# In[ ]:


start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end).rename('SARIMAX(2, 0, 2)x(2, 1, [], 12) Predictions')


# In[ ]:


test.plot(legend=True,figsize=(12,6))
predictions.plot(legend=True)


# In[ ]:


from sklearn.metrics import mean_squared_error

error = mean_squared_error(test, predictions)
print('Mean Squared Error ',error)


# In[ ]:


from statsmodels.tools.eval_measures import rmse

error = rmse(test, predictions)
print('Root Mean Squared Error ',error)


# # Training with the entire Dataset

# In[ ]:


model = SARIMAX(resample_monthly_temp,order=(2,0,2), seasonal_order=(2,1,0,12), m=12)
results = model.fit()
fcast = results.predict(len(resample_monthly_temp)-1,len(resample_monthly_temp)+11,typ='levels').rename('SARIMA(2,0,2)(2,1,0,12) Forecast')


# # Forecasted Values

# In[ ]:


fcast


# # Visualization of Forecast Data along with Historical Data

# In[ ]:


resample_monthly_temp.plot(legend=True,figsize=(12,6))
fcast.plot(legend=True)

