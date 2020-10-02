#!/usr/bin/env python
# coding: utf-8

# ![Introduction](https://i.imgur.com/gMMzZ1K.png)
# 
# ## Introduction
# 
# Recently I got interested in stock markets and I wanted to get some deeper understanding of how it all works. Usually the very first thing that comes to your mind when you are thinking of stocks is a price chart. You can see how the price changes over time for a specific stock. However, there is a lot of other important charts or statistics that may improve your understanding of stock's behavior e.g. its trend or momentum. It may also help you to try to predict future prices (up to some accuracy).
# 
# In this notebook I will explore history of SPY exchange traded fund. I will introduce and visualize few technical indicators that are commonly used in technical analysis. I am planning to use those to predict stock trend or future prices in another kernel.

# In[ ]:


import os
import pandas as pd

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Show charts when running kernel
init_notebook_mode(connected=True)

# Change default background color for all visualizations
layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(250,250,250,0.8)')
fig = go.Figure(layout=layout)
templated_fig = pio.to_templated(fig)
pio.templates['my_template'] = templated_fig.layout.template
pio.templates.default = 'my_template'

def plot_scatter(x, y, title):
    fig = go.Figure(go.Scatter(x=x, y=y, name=title))
    fig.update_layout(title_text=title)
    fig.show()


# ## Historical ETF prices
# 
# Data frame with historial prices for SPY fund consists of 3201 rows, each with 7 columns which are: *Date*, *Open/High/Low/Close* prices, *Volume* count and *Open Interest* number. *OpenInt* column has only 0 values, so I will just ignore it and focus on the rest of information. In tables below you can see sample prices from the data frame and also few statistics about each column e.g. min/max values, standard deviation etc.

# In[ ]:


ETF_NAME = 'SPY'
ETF_DIRECTORY = '/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/ETFs/'

df = pd.read_csv(os.path.join(ETF_DIRECTORY, ETF_NAME.lower() + '.us.txt'), sep=',')
df.head()


# In[ ]:


df.describe()


# ## I. OHLC Chart
# 
# An OHLC chart shows the *open*, *high*, *low* and *close* prices of a stock. It shows you how the price was changing during a particular day and give you a sense of e.g. momentum or volatility of stock. The tip of the lines represent the *low* and *high* values and the horizontal segments represent the *open* and *close* values. Sample points where the close value is higher (lower) then the open value are called increasing (decreasing). By default, increasing items are drawn in green whereas decreasing are drawn in red.

# In[ ]:


fig = go.Figure([go.Ohlc(x=df.Date,
                         open=df.Open,
                         high=df.High,
                         low=df.Low,
                         close=df.Close)])
fig.update(layout_xaxis_rangeslider_visible=False)
fig.show()


# ## II. Volume
# 
# A [*volume*](https://www.investopedia.com/terms/v/volume.asp) is a very basic measure that shows a number of shares traded (bought, sold) over a certain period of time e.g. daily. It is such a simple but often overlooked indicator. Volume is so important because it basically represents the activity in stock trading. Higher volume value indicates higher interests in trading a stock.

# In[ ]:


fig = go.Figure(go.Bar(x=df.Date, y=df.Volume, name='Volume', marker_color='red'))
fig.show()


# ## III. Moving Averages
# 
# [Moving Averages](https://www.investopedia.com/terms/m/movingaverage.asp) (MA) help to smooth out stock prices on a chart by filtering out short-term price fluctuations. We calculate moving averages over a defined period of time e.g. last 9, 50 or 200 days. There are two (most common) averages used in technical analysis which are:
# * *Simple Moving Average (SMA)* - a simple average calculated over last N days e.g. 50, 100 or 200,
# * *Exponential Moving Average (EMA)* - an average where greater weights are applied to recent prices.
# 
# MAs and their crossovers (see [Golden Cross](https://www.investopedia.com/terms/g/goldencross.asp) and [Death Cross](https://www.investopedia.com/terms/d/deathcross.asp) are often used as trade signals as they are so simple yet powerful.

# In[ ]:


df['EMA_9'] = df['Close'].ewm(5).mean().shift()
df['SMA_50'] = df['Close'].rolling(50).mean().shift()
df['SMA_100'] = df['Close'].rolling(100).mean().shift()
df['SMA_200'] = df['Close'].rolling(200).mean().shift()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.Date, y=df.EMA_9, name='EMA 9'))
fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_50, name='SMA 50'))
fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_100, name='SMA 100'))
fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_200, name='SMA 200'))
fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close', line_color='dimgray', opacity=0.3))
fig.show()


# ## IV. RSI
# 
# Another commonly used indicator is a [Relative Strength Index](https://www.investopedia.com/terms/r/rsi.asp) (RSI) that indicates magnitude of recent price changes. It can show that a stock is either *overbought* or *oversold*. Typically RSI value of 70 and above signal that a stock is becoming overbought/overvalued, meanwhile value of 30 and less can mean that it is oversold. Full range of RSI is from 0 to 100.

# In[ ]:


def RSI(df, n=14):
    close = df['Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

num_days = 365
df['RSI'] = RSI(df).fillna(0)
fig = go.Figure(go.Scatter(x=df.Date.tail(num_days), y=df.RSI.tail(num_days)))
fig.show()


# ## V. MACD
# 
# [Moving Average Convergence Divergence](https://www.investopedia.com/terms/m/macd.asp) (MACD) is an indicator which shows the relationship between two exponential moving averages i.e. 12-day and 26-day EMAs. We obtain MACD by substracting 26-day EMA (also called *slow EMA*) from the 12-day EMA (or *fast EMA*).
# 
# You can more more about entry/exit signals that can be read from MACD under [this link](https://www.investopedia.com/terms/m/macd.asp).

# In[ ]:


EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
MACD = pd.Series(EMA_12 - EMA_26)
MACD_signal = pd.Series(MACD.ewm(span=9, min_periods=9).mean())

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.Date, y=EMA_12, name='EMA 12'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.Date, y=EMA_26, name='EMA 26'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.Date, y=MACD, name='MACD'), row=2, col=1)
fig.add_trace(go.Scatter(x=df.Date, y=MACD_signal, name='Signal line'), row=2, col=1)
fig.show()


# ## VI. Stochastic
# 
# The last technical tool in this notebook is a [stochastic oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp) is quite similar to RSI in the sense that it's values (also in range 0-100) can indicate whether a stock is overbought/oversold or not. It is arguably the most complicated indicator compared to the ones introduced earlier. Stochastic can be calculated as:
# 
# $$\%K = (\frac{C - L_{14}}{H_{14} - L_{14}}) \times 100$$
# 
# where: $C$ is the most recent close price, $L_{14}$ and $H_{14}$ are the lowest/highest prices traded in last 14 days.
# 
# This $\%K$ stochastic is often referred as the *"slow stochastic indicator"*. There is also a *"fast stochastic indicator"* that can be obtained as:
# 
# $$\%D = SMA_{3}(\%K)$$
# 
# You can read more about stochastics and how their values/crossovers indicate trade signals [here](https://www.investopedia.com/terms/s/stochasticoscillator.asp).

# In[ ]:


# https://stackoverflow.com/questions/30261541/slow-stochastic-implementation-in-python-pandas
def stochastic(df, k, d):
    df = df.copy()
    low_min  = df['Low'].rolling(window=k).min()
    high_max = df['High'].rolling( window=k).max()
    df['stoch_k'] = 100 * (df['Close'] - low_min)/(high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(window=d).mean()
    return df

stochs = stochastic(df, k=14, d=3)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.Date.tail(365), y=stochs.stoch_k.tail(365), name='K stochastic'))
fig.add_trace(go.Scatter(x=df.Date.tail(365), y=stochs.stoch_d.tail(365), name='D stochastic'))
fig.show()


# That's all in this notebook. However, there are a lot of other tools and indicators for *technical analysis* of stocks. Based on individual trader's strategy and goals, different indicators are being used for analysis. Moreover, analyzing averages, momentum and trends is not enough to forecast future prices. Combining them with *fundamental analysis* and monitoring news (e.g. Tweets) can increase your chances to predict prices correctly.
