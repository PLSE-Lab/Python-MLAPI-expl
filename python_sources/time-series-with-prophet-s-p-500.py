#!/usr/bin/env python
# coding: utf-8

# ## Using Prophet for Time-Series Data
# 
# ### Introduction:
# In this kernel we will predict stock prices using Prophet, Facebook's library for time series prediction. 
# 
# ### About Prophet:
# Prophet is Facebook's way to make time-series prediction simpler. 
# 
# Prophet is used across Facebook for goal setting and planning.
# 
# It can be used effectively in the following situations:
# 
# 1. Non-linear trends with yearly, weekly and daily seasonality
# 2. Time series with strong seasonal effect
# 3. Datasets with several seasons of historical data
# 
# It is a useful tool since the syntax is very similar to scikit-learn methods and can handle misssing data and outliers well.
# 
# More about Prophet: https://facebook.github.io/prophet/

# ### Contents:
# 
# ### <a href='#stock-charts'>1. Stock Charts </a>
#     
# ### <a href='#prophet'>2. Time Series Analysis with Prophet</a>
# 
# ### <a href='#references'>3. References </a>

# In[ ]:


get_ipython().system('pip3 uninstall --yes fbprophet')
get_ipython().system('pip3 install fbprophet --no-cache-dir --no-binary :all:')


# In[ ]:


import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from scipy.stats import norm
import datetime as dt
from fbprophet import Prophet


# In[ ]:


all_stocks = pd.read_csv('../input/all_stocks_5yr.csv', index_col = 'date', parse_dates=['date'])
all_stocks.head()


# In[ ]:


apple = all_stocks[all_stocks['Name'] == 'AAPL']
apple.head()


# In[ ]:


apple.info()


# In[ ]:


apple.describe()


# <a id='stock-charts'></a>

# ### 1. Stock Charts

# In[ ]:


apple['2014':'2017'].plot(subplots=True, figsize=(10, 12))
plt.title('Apple stock attributes from 2014 to 2017')


# In[ ]:


amazon = all_stocks[all_stocks['Name'] == 'AMZN']
amazon.head()


# #### Comparing stock prices of Apple and Amazon (not normalized prices)

# In[ ]:


apple.high.plot()
amazon.high.plot()
plt.legend(['Apple', 'Amazon'])


# #### Normalizing the stock prices 

# In[ ]:


norm_apple = apple.high.div(apple.high.iloc[0]).mul(100)
norm_amazon = amazon.high.div(amazon.high.iloc[0]).mul(100)
norm_apple.plot()
norm_amazon.plot()
plt.legend(['Apple', 'Amazon'])


# ### OHLC and Candlestick Chart:
# OHLC, or open-high-low-close, charts are a type of bar chart that shows open, high, low, and closing prices. Unlike line charts, OHLC charts enable technical analysts to evaluate intraday volatility and see where prices opened and closed. 
# (https://www.investopedia.com/terms/o/ohlcchart.asp)
# 
# Traders use the candlesticks to make trading decisions based on regularly occurring patterns that help forecast the short-term direction of the price.
# (https://www.investopedia.com/trading/candlestick-charting-what-is-it/)
# 
# 
# <img src="https://mr-uploads.s3.amazonaws.com/uploads/2014/11/bar-and-candle-1.png">

# #### OHLC Chart for Apple

# In[ ]:


trace = go.Ohlc(x = apple.index,
               open = apple['open'],
               high = apple['high'],
               low = apple['low'],
               close = apple['close'])

layout = {
    'xaxis': {
        'title':'Date',
        'rangeslider':{'visible':False}
    },
    'yaxis': {
        'title':'Price in US Dollars'
    },
    'shapes': [{
        'x0': '2016-11-08', 'x1': '2016-11-08',
        'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',
        'line': {'color': 'rgb(30,30,30)', 'width': 1}
    }],
    'annotations': [{
        'x':'2016-11-08', 'y':0.05, 'xref':'x', 'yref':'paper', 'showarrow':False, 'text': 'US Presidential Election'
    }]
}

data = [trace]
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


apple['close_std'] = apple['close'].rolling(5).std()
apple['close_mean'] = apple['close'].rolling(5).mean()

fig, (std, avg) = plt.subplots(1, 2, figsize=(16, 8))
std.plot(apple.index, apple['close_std'], label='5 day Standard Deviation')
std.legend()

avg.plot(apple.index, apple['close_mean'], label='5 Day Moving Average', color='green')
avg.plot(apple.index, apple['close'].rolling(20).mean(), label='20 Day Moving Average', color='red')
avg.legend()


# In[ ]:


apple_two = apple['2016':'2018']
apple_two = apple_two.reset_index()
apple_two.head()


# In[ ]:


apple_monthly = apple_two.groupby(['date'])


# In[ ]:


agg = apple_monthly.aggregate({'open':np.mean, 'high':np.mean, 'low':np.mean, 'close':np.mean})
agg = agg.reset_index()
agg.head()


# In[ ]:


trace = go.Candlestick(x = agg['date'],
                      open = agg['open'].values.tolist(),
                      high = agg['high'].values.tolist(),
                      low = agg['low'].values.tolist(),
                      close = agg['close'].values.tolist()
                      )
layout = {
    'title':'Apple Stock from 2016 to 2018',
    'xaxis': {'title':'Date',
             'rangeslider':{'visible':False}},
    'yaxis':{'title':'Price in US Dollars'}
}


data = [trace]


fig_candle = go.Figure(data, layout)
iplot(fig_candle)


# The Apple stock price goes down in January. Let us look at that closely.

# In[ ]:


apple_2018 = apple['2018']
apple_2018 = apple_2018.reset_index()
apple_2018.head()


# In[ ]:


apple_m = apple_2018.groupby(['date'])
agg_m = apple_m.aggregate({'open':np.mean, 'close':np.mean, 'high':np.mean, 'low':np.mean})
agg_m = agg_m.reset_index()

trace = go.Candlestick(x = agg_m['date'],
                       open = agg_m['open'],
                       high = agg_m['high'],
                       low = agg_m['low'],
                       close = agg_m['close']
                      )

data = [trace]

layout = {
    'title':'Closer look at Apple Stock for downward trend',
    'xaxis': {'title':'Date',
             'rangeslider':{'visible':False}},
    'yaxis':{'title':'Price in US Dollars'}
}

fig_candle1 = go.Figure(data=data, layout=layout)
iplot(fig_candle1)


# <a id="prophet"></a>

# ### Time Series with Prophet 

# Prophet works best with datasets with high degree of seasonality, however, my aim with this is kernel is to use it for simple forecasting and show how simple it is to use Prophet.
# 
# #### Data manipulation for Prophet:
# 
# 1. input is always a dataframe with two columns 'y' and 'ds'.
# 2. y is the numeric quantity you want to forecast
# 3. ds is the datestamp column and accepted dateformat is YYYY-MM-DD
# 

# In[ ]:


apple.head()


# In[ ]:


model = Prophet()
ph_apple = apple.drop(['open', 'high', 'low', 'volume', 'Name', 'close_std', 'close_mean'], axis=1)

ph_apple = ph_apple.reset_index()


# In[ ]:


ph_apple.head()


# In[ ]:


ph_apple = ph_apple.rename(columns = {'close':'y', 'date':'ds'})


# In[ ]:


model.fit(ph_apple)


# In[ ]:


future = model.make_future_dataframe(periods=365)
future.tail()


# In[ ]:


forecast = model.predict(future)


# In[ ]:


forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].tail()


# In[ ]:


fig1 = model.plot(forecast)


# We can also analyze the points where the trend changes by calling the add_changepoints_to_plot method.
# 
# By default changepoints are only inferred for the first 80% of the time series in order to have plenty of runway for projecting the trend forward and to avoid overfitting fluctuations at the end of the time series. 

# In[ ]:


from fbprophet.plot import add_changepoints_to_plot
fig1 = model.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), model, forecast)


# In[ ]:


fig2 = model.plot_components(forecast)


# ### Insights:
#    - There is an upward trend for the stock price since a dip in mid-2016. 
#    - There is no weekly trend 
#    - During June, Apple stock increases since June is the time for the Apple WWDC. There is a minor increase in August-September also, probably due to anticipation of the new iPhones. There is a spike again in October-November, since Apple launches its new iPhones in that timeframe. Also, Black Friday is in November and that gives a boost to Apple product sales.
#    - Stock price goes down in January-February. 

# ### Monthly Prediction

# In[ ]:


model = Prophet(changepoint_prior_scale=0.05).fit(ph_apple)
future_m = model.make_future_dataframe(periods=12, freq='M')
m_forecast = model.predict(future_m)
fig1 = model.plot(m_forecast)
plt.title('Monthly Predictions (1 year timeframe)')


# In[ ]:


fig2 = model.plot_components(m_forecast)


# <a id='references'></a>

# ### References
# 
# 1. https://www.kaggle.com/thebrownviking20/everything-you-can-do-with-a-time-series
# 2. https://www.kaggle.com/janiobachmann/s-p-500-time-series-forecasting-with-prophet/data
