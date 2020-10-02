#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = (15, 6)


# In[ ]:


df = pd.read_csv('../input/international-airline-passengers.csv')


# In[ ]:


df.head()


# In[ ]:


df = df.drop(df.index[144])


# In[ ]:


df['Month'] = pd.to_datetime(df['Month'],yearfirst=True)


# In[ ]:


df.info()


# In[ ]:


df.index = df['Month']


# In[ ]:


df = df.drop('Month',axis=1)


# In[ ]:


df.columns = ['Passengers']


# In[ ]:


df.head()


# In[ ]:


df.plot()


# In[ ]:


from statsmodels.tsa.stattools import adfuller as adf
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    


# In[ ]:


test_stationarity(df)


# In[ ]:


output = (adf(df['Passengers']))


# In[ ]:


output


# In[ ]:


dfoutput = pd.Series(output[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in output[4].items():
        dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# As p-value is quite greater than critical value, there is non stationarity which can be seen from an increasing rolling mean

# In[ ]:


#estimating trend and seasonlity
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(df, label='Original')
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


#we will first use arimax to account only for trend and then later will use sarimax to account for both trend and seasonality


# In[ ]:


#divide into train and validation set
train = df[:int(0.7*(len(df)))]
valid = df[int(0.7*(len(df))):]

#plotting the data
ax = train.plot()
valid.plot(ax=ax)


# In[ ]:


#building the model
from pmdarima.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.legend(loc='best')
plt.show()


# In[ ]:


#using sarimax to account for seasonality and then forecasting
#building the model
from pmdarima.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True,seasonal=True,m=12,D=1)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.legend(loc='best')
plt.show()


# In[ ]:


#will add in more theory to explain better

