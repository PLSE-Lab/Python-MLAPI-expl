#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import statsmodels
import numpy as np
import pandas as pd 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from matplotlib import pyplot
from numpy import sqrt,mean,log,diff
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[ ]:


pip install pyramid-arima


# In[ ]:


from pyramid.arima import auto_arima


# In[ ]:


ggl = pd.read_csv("/kaggle/input/amdgoogle/GOOGL.csv")


# In[ ]:


ggl.tail(5)


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=ggl.Date, y=ggl['Open'], name="Open",
                         line_color='deepskyblue'))

fig.add_trace(go.Scatter(x=ggl.Date, y=ggl['High'], name="High",
                         line_color='dimgray'))

fig.add_trace(go.Scatter(x=ggl.Date, y=ggl['Low'], name="Low",
                         line_color='royalblue'))

fig.add_trace(go.Scatter(x=ggl.Date, y=ggl['Close'], name="Close",
                         line_color='firebrick'))

fig.update_layout(title_text='Time Series with Rangeslider',
                  xaxis_rangeslider_visible=True)
fig.show()


# **All four channels shown above are close to each other. We can take any one of them in forecasting Analysis. ** 
# 
# **Also the above series is not stationary as the trendline is keep on increasing **
# We can check for the Seasonality in the Curve 

# In[ ]:


TimeSeries = pd.DataFrame(ggl, columns=["Date","Open","Volume"])


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(
                x=TimeSeries.Date,
                y=TimeSeries.Volume,
                name="Volume in Time-Stamp",
                line_color='deepskyblue',
                opacity=0.8))

fig.update_layout(xaxis_range=['2009-05-22','2018-08-29'],
                  title_text="Volume on Time-Stamp")
fig.show()


# The seasonal effects are usually adjusted so that they average to 0 for an additive decomposition or they average to 1 for a multiplicative decomposition. In our case the seasonal variation is slightly increasing over time so multiplicative model is prefered in this case. 
# 
# We can decompose our Time Series into three components.
# 
# 1. Trend Component(T(t))
# 2. Seasonal Component(S(t))
# 3. Residual/ Noise/ Irregular Component(R(t))
# 
# Aditive Model: Y(t) = T(t) + S(t) + R(t)
# 
# Multiplicative Model: Y(t) = T(t) x S(t) x R(t)
# 
# Where Y(t) is our Original Time Series.

# In[ ]:


# result = seasonal_decompose(TimeSeries['Open'].values, model='multiplicative', extrapolate_trend='freq', freq=365)
# result.plot()
# plt.show();

plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':120})

decomposition = seasonal_decompose(TimeSeries['Open'].values, model='multiplicative', extrapolate_trend='freq', freq=365)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(TimeSeries['Open'].values, label='Original')
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


# There is a Seasonality Factor which is repeating on an yearly basis. Trend is approximately Linear and Increasing. Overriding residuals can be seen separately in the above figure. 

# differencing is required only when the series is non-stationary. Else, no differencing is needed, that is if, d=0.
# 
# The null hypothesis of the ADF test is that the time series is non-stationary. So, if the p-value of the test is less than the significance level (0.05) then you reject the null hypothesis and infer that the time series is indeed stationary.
# 
# So, in our case, if P Value > 0.05 we go ahead with finding the order of differencing.

# In[ ]:


Test = np.array(TimeSeries.Open)
result = adfuller(Test)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[ ]:


plt.rcParams.update({'figure.figsize':(10,5), 'figure.dpi':120})
df = log(Test)
# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df); axes[0, 0].set_title('Original Series')
plot_acf(df, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(diff(df)); axes[1, 0].set_title('1st Order Differencing')
plot_acf(diff(df), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(diff(diff(df))); axes[2, 0].set_title('2nd Order Differencing')
plot_acf( diff(diff(df)), ax=axes[2, 1])

plt.show()


# ** One Differencing is enough to make the series stationary.**

# In[ ]:


plt.rcParams.update({'figure.figsize':(8,2), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(diff(df)); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(diff(df), ax=axes[1])

plt.show()


# In[ ]:


plt.rcParams.update({'figure.figsize':(8,2), 'figure.dpi':120})
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(diff(df)); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(diff(df), ax=axes[1])


# In[ ]:


data = pd.DataFrame(ggl, columns=["Date","Open"])


# In[ ]:


data.Date = pd.to_datetime(data.Date)
data = data.set_index('Date')
data.head()


# In[ ]:


train = data.loc['2009-05-22':'2016-12-01']
test = data.loc['2016-12-02':]


# In[ ]:


plt.rcParams.update({'figure.figsize':(10,4), 'figure.dpi':120})
train['Open'].plot()
test['Open'].plot();


# In[ ]:


# model = ARIMA(df, order=(1,2,1))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())


# In[ ]:


# results= model_fit.plot_predict(dynamic=False)
# plt.show()


# In[ ]:


# stepwise_model = auto_arima(df, start_p=1, start_q=1,
#                            max_p=3, max_q=3, m=12,
#                            start_P=0, seasonal=True,
#                            d=1, D=1, trace=True,
#                            error_action='ignore',  
#                            suppress_warnings=True, 
#                            stepwise=True)
# print(stepwise_model.aic())


# In[ ]:


# stepwise_model.fit(train)


# In[ ]:


# future_forecast = stepwise_model.predict(n_periods=len(test))


# In[ ]:


# future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=["Open_test"])
# pd.concat([test,future_forecast],axis=1).plot();


# In[ ]:


# future_forecast = pd.DataFrame(future_forecast,columns=["Open"])
# future_forecast.plot()
# plt.rcParams.update({'figure.figsize':(10,4), 'figure.dpi':120})
# train['Open'].plot()
# test['Open'].plot();


# In[ ]:


model = auto_arima(df, start_p=3, start_q=3,
                           max_p=5, max_q=5, m=25,
                           start_P=1, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
model.fit(train)


# In[ ]:


forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Open_test'])

plt.plot(train, label='Train')
plt.plot(test, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()

