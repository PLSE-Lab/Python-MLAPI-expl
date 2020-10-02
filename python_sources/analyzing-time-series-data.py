#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Time series is a sequence of observations recorded at regular time intervals.
# 
# Time series analysis is a statistical technique to analyze the pattern of data points taken over time to forecast the future. In this kernel, we will see major components or pattern that are analyzed through time series data.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from plotly.offline import init_notebook_mode, iplot

import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

# Few libraries for visualizing time series data
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[ ]:


data = pd.read_csv("../input/nifty50-stock-market-data/COALINDIA.csv")
data.set_index("Date", drop=False, inplace=True)


# In[ ]:


data.head()


# In[ ]:


# check missing/null values

msno.matrix(data)


# First we will see how our target feature relate with the dates.

# In[ ]:


fig = px.line(data, x="Date", y="VWAP")
py.iplot(fig, filename="simple_line")


# It is increasing at some point and decreasing at some.
# 
# Let's talk about components.
# * **Trends**
# 
# Increase or decrease in the series of data over a longer period.
# 
# * **Seasonality**
# 
# Fluctuations in the pattern due to seasonal determinants over a short period. 
# 
# * **Cyclicity** 
# 
# Variations occurring at irregular intervals due to certain circumstances. 
# 
# * **Irregularity (or noise)**
# 
# Instability due to random factors that do not repeat in the pattern. It represents the residuals or remainder of the time series after the other components have been removed.
# 
# Sometimes the trend and cyclical components are grouped into one, called the trend-cycle component. The trend-cycle component can just be referred to as the "trend" component, even though it may contain cyclical behavior.

# We will decompose a time series into its components using `seasonal_decompose` in `statsmodel`.

# In[ ]:


# Let's take CoalIndia stocks High price for this
data1 = pd.read_csv("../input/nifty50-stock-market-data/COALINDIA.csv", index_col="Date", parse_dates=["Date"])

Plot = seasonal_decompose(data1["High"], freq=360)
plt.rcParams.update({'figure.figsize' : (10, 10)})
Plot.plot()
plt.show()


# * There is clearly a downward trend 
# * There is a uniform seasonal change in the above plot
# * Residuals are non-uniform which might be an indication of outliers and missing values.

# ## Finance
# 
# let's visualize further:
# 
# **OHLC Charts**
# 
# An OHLC chart is a type of price chart that shows the open, high, low and close price of a certain period. Open-High-Low-Close Charts (OHLC Charts) are used as a trading tool to visualize and analyze the price changes over time for stocks, bonds etc.

# In[ ]:


fig = go.Figure(data=[go.Ohlc(
    x=data.index,
    open=data.Open, high=data.High,
    low=data.Low, close=data.Close,
    increasing_line_color= 'cyan', decreasing_line_color= 'gray'
)])
py.iplot(fig, filename="simple_ohlc")


# The tip of the lines represent the low and high values and the horizontal segments represent the open and close values. Sample points where the close value is higher (lower) then the open value are called increasing (decreasing).

# **Candlestick Charts**
# 
# Candlestick chart is used as a trading tool to visualise and analyse the price movements over time for securities, derivatives, currencies, stocks, bonds, commodities, etc. Although the symbols used in Candlestick Charts resemble a Box Plot, they function differently and therefore, are not to be confused with one another.

# In[ ]:


fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data.Open, high=data.High,
    low=data.Low, close=data.Close,)])
py.iplot(fig, filename="simple_candlestick")


# The boxes represent the spread between the open and close values and the lines represent the spread between the high and low values. Sample points where the close value is higher (lower) than the open value are called increasing (decreasing).

# ### Autocorrelation and Partial Autocorrelation
# 
# - **Autocorrelation (acf)** measures how a series is correlated with itself at different lags. If a series is significantly autocorrelated, that means, the previous values of the series (lags) may be helpful in predicting the current value. 

# In[ ]:


plot_acf(data.VWAP, lags=50, title="VWAP")
plt.show()


# Lags are only decreasing after first lag.
# 
# - **Partial Autocorrelation (pacf)** can be interpreted as a regression of the series against its past lags. Partial Autocorrelation also conveys similar information but it conveys the pure correlation of a series and its lag, excluding the correlation contributions from the intermediate lags. 

# In[ ]:


plot_pacf(data.VWAP, lags=50, title="VWAP")
plt.show()


# Partial Autocorrelation after first 2 lags is very low. There are few lags which are not statistically significant.

# **If you like this kernel please upvote it or if you have any queries or suggestions then leave a comment.**
