#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# In this kernel I will deliver EDA and Forecast for Stock Prices from Spanish Stocks Historical Data. <br>
# Dataset which will be primary focus of the work is about the company named <b>Repsol S.A</b>
# 
# Repsol is Spanish energy company that hires over 24,000 of employees world-wide. It operates in many areas of the oil and gas industry among which are exploration, production, refining and distribution of petrochemicals.

# ## Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns

import plotly
import plotly.plotly as py
import cufflinks as cf
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()

from fbprophet import Prophet

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Data Import and Analysis

# In[ ]:


df = pd.read_csv("../input/repsol.csv")
df.head(10)


# In[ ]:


df.info()


# In[ ]:



trace_open = go.Scatter(
    x = df["Date"],
    y = df["Open"],
    mode = 'lines',
    name="Open"
)

trace_high = go.Scatter(
    x = df["Date"],
    y = df["High"],
    mode = 'lines',
    name="High"
)

trace_low = go.Scatter(
    x = df["Date"],
    y = df["Low"],
    mode = 'lines',
    name="Low"
)

trace_close = go.Scatter(
    x = df["Date"],
    y = df["Close"],
    mode = 'lines',
    name="Close"
)



data = [trace_open,trace_high,trace_low,trace_close]

layout = go.Layout(title="Repsol Stock Price",xaxis_rangeslider_visible=True)

fig = go.Figure(data=data,layout=layout)

plotly.offline.iplot(fig)


# In[ ]:


trace_volume = go.Scatter(
    x = df["Date"],
    y = df["Volume"],
    mode = 'lines',
    name="Volume"
)

data_volume = [trace_volume]

layout_volume = go.Layout(title="Volume",xaxis_rangeslider_visible=True)

fig_volume = go.Figure(data=data_volume,layout=layout_volume)

plotly.offline.iplot(fig_volume)


# As can be seen in Volume chart, either Volume of Repsol stock reached 0 or we have no information about volume of that stock. In that case I will slice data to only get dataset from Dec 2010 to 2019.

# In[ ]:


df.shape


# In[ ]:


df_fc = df[935:3140] # 2010-10-21 date
df_fc.head()


# ## Forecast
# 
# Forecasting will be done using Facebook's library called Prophet which was designed for time series forecasting and on top of that it is using Sklearn API so it should be quite easy to understand.
# 
# Prophet takes data as a dataframe and it use two main labels. One is "ds" as datastamp and another is "y" as value that we want to forecast.
# After we fit dataframe to model we make a forecast. As a result we are given dataframe with predicted values. Among those values are the boundaries of the uncertainty interval (lower and upper).
# 

# In[ ]:


model = Prophet()


df_prophet = df_fc.drop(['Open', 'High', 'Low','Volume'], axis=1)
df_prophet.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

df_prophet.head(10)


# In[ ]:


model.fit(df_prophet)


# In[ ]:



future_prices = model.make_future_dataframe(periods=365)
forecast = model.predict(future_prices)
df_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
df_forecast.head()


# In[ ]:


trace_open = go.Scatter(
    x = df_forecast["ds"],
    y = df_forecast["yhat"],
    mode = 'lines',
    name="Forecast"
)

trace_high = go.Scatter(
    x = df_forecast["ds"],
    y = df_forecast["yhat_upper"],
    mode = 'lines',
    fill = "tonexty", 
    line = {"color": "#57b8ff"}, 
    name="Higher uncertainty interval"
)

trace_low = go.Scatter(
    x = df_forecast["ds"],
    y = df_forecast["yhat_lower"],
    mode = 'lines',
    fill = "tonexty", 
    line = {"color": "#57b8ff"}, 
    name="Lower uncertainty interval"
)

trace_close = go.Scatter(
    x = df_prophet["ds"],
    y = df_prophet["y"],
    name="Data values"
)



data = [trace_open,trace_high,trace_low,trace_close]

layout = go.Layout(title="Repsol Stock Price Forecast",xaxis_rangeslider_visible=True)

fig = go.Figure(data=data,layout=layout)

plotly.offline.iplot(fig)


# ## Historical Price
# 
# ### Candlesticks 
# Historical Price chart are often represented in candlestick charts which are used to show traders "emotion" by visually representing the size of price moves with two different colors. Traders are using those charts to to make trading decisions based on occurring patterns.
# 
# There are two main patterns that are used in candlestick charts. One is bullish and another one is bearish. 
# Bullish patters indicate that price of the stock is likely to rise, while Bearish patter indicate fall in the stock price. However one must remember that candlestick charts and its patterns are representing only tendencies in price movement.
# 
# Here's image I made that explains Candlestick's
# <img src="https://i.ibb.co/BztW98W/Bearish-Redish.png" alt="Bearish-Redish" border="0" />
# 
# 
# 

# In[ ]:





trace = go.Candlestick(x = df_fc['Date'].values.tolist(),
                       open = df_fc['Open'].values.tolist(),
                       high = df_fc['High'].values.tolist(),
                       low = df_fc['Low'].values.tolist(),
                       close = df_fc['Close'].values.tolist(),
                      increasing=dict(line=dict(color= '#58FA58')),
                decreasing=dict(line=dict(color= '#FA5858')))

layout = go.Layout(title="Repsol Historical Price",
                   xaxis_rangeslider_visible=False,
                   yaxis= {'title': 'Stock Price'}
                  )




data = [trace]

fig = go.Figure(data=data,layout=layout)

plotly.offline.iplot(fig)


# ## Summary
# 
# This is first financial kernel that I have done and I hope everyone enjoyed it. In case of any advice, question or finding any errors feel free to post comment down below or message me directly. 
# Keep in mind this is still work in progres.
# 
# Have a nice Kaggling everyone :) 

# In[ ]:




