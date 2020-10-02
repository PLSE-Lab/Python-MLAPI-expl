#!/usr/bin/env python
# coding: utf-8

# **Two Sigma Financial News Competition**
#    
#    Let's import the data set.

# In[ ]:


import pandas as pd
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools
from datetime import datetime 
import math
import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# In[ ]:


market_train_df.head()


# In[ ]:


market_train_df.tail()


# Market data provided by contains financial market information such as opening price, closing price, trading volume, calculated returns(various).
# 1.  Let's store each assetname in assetName so that we can asses the individually.
# 2.  We'll use index to call each asset.

# In[ ]:


#store the asset names in 'assetNames' to view the stock prices individually
assetNames = market_train_df.assetName.unique()
assetCodes = market_train_df.assetCode.unique()
print("Number of assets by asset names :" , len(assetNames))
print("Number of assets by asset codes :" , len(assetCodes))
print("Asset name at index 0 : " ,assetNames[0])
print("Asset name at index 1 : " ,assetNames[1])
print("Asset name at index 100 : " ,assetNames[100])
print("Asset name at index 1000 : " ,assetNames[1000])


# For a particular asset (stock) we have:
# 1. assetCode(object) - a unique id of an asset
# 2. assetName(category) - the name that corresponds to a group of assetCodes. 
# 
# > assetName may be "Unknown" if the corresponding assetCode does not have any rows in the news data.
# > As given in data description
# 
# Therefore, few companies have no assetName and are listed as 'Unknown'.
# We'll look into the data by asset names as it is the company name.

# In[ ]:


#function to subset market data by particular asset name or asset code
def subset(assetname = "", by_code = False, assetcode = '' ):
    if by_code == True:
        data = market_train_df[market_train_df.assetCode == assetcode]
    else:
        data = market_train_df[market_train_df.assetName == assetname]            
    return data


# In[ ]:


#layout for plotting finalcial charts
#since its a time series data from 2007 to 2017 we'll use 
#rangeselectors and sliders to view charts
layout = dict(
            xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label='1m',
                         step='month',
                         stepmode='backward'),
                    dict(count=3,
                         label='3m',
                         step='month',
                         stepmode='backward'),
                    dict(count=6,
                         label='5m',
                         step='month',
                         stepmode='backward'),
                    dict(count=1,
                         label='1y',
                         step='year',
                         stepmode='backward'),
                    dict(count=5,
                         label='5y',
                         step='year',
                         stepmode='backward'),
                    dict(label ='max',
                         step='all')
                ])
            ),
            rangeslider=dict(
                visible = True
            ),
            type='date'
        )
    )


# 

# **Simple Moving Average**
# One of the oldest and simplest trading strategies that exist is the one that uses a moving average of the price (or returns) timeseries to proxy the recent trend of the price.
# A simple moving average (SMA )is an arithmetic moving average calculated by adding recent closing prices and then dividing that by the number of time periods in the calculation average. A simple, or arithmetic, moving average that is calculated by adding the closing price of the security for a number of time periods and then dividing this total by that same number of periods. Short-term averages respond quickly to changes in the price of the underlying, while long-term averages are slow to react.
# 
# Investopedia https://www.investopedia.com/terms/s/sma.asp#ixzz5SgMe6TFa
# 
# 

# In[ ]:


def moving_avg(data ,period=0, column='open' ):
    data['sma' + str(period)] = data[column].rolling(window = period).mean()
    return data


# **Exponential Moving Average**
# The exponential moving average (EMA) is a weighted moving average (WMA) that gives more weighting, or importance, to recent price data than the simple moving average (SMA) does.
# 
# Investopedia https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp#ixzz5SgMPeeN3

# In[ ]:


def ema(data, period=0, column='open'):
    data['ema' + str(period)] = data[column].ewm(ignore_na=False,
                                                 min_periods=period,
                                                 com=period, adjust=True).mean()
    
    return data


# **Bollinger Bands**
# Bollinger Bands are a type of statistical chart characterizing the prices and volatility over time of a financial instrument or commodity,
# Bollinger Bands indicator is popular among traders across several financial markets. On a chart, Bollinger Bands are two "bands" that sandwich the market price. Many traders use them primarily to determine overbought and oversold levels.
# 
# Investopedia https://www.investopedia.com/walkthrough/forex/intermediate/level4/bollinger-bands.aspx#ixzz5SgN9i6ZQ
# 

# In[ ]:


def bollinger_bands(data, periods=20, close_col='open'):
    data['MA'] = data[close_col].rolling(window=periods).mean()
    data['STD'] = data[close_col].rolling(window=periods).std()
    data['UpperBand'] = data['MA'] + (data['STD'] * 2)
    data['LowerBand'] = data['MA'] - (data['STD'] * 2)
    data = data.drop(['MA','STD'], axis = 1)
    return data


# In[ ]:


def make_plot(layout: dict,    #base layout
              period: int,     #period over which indicators are to be calculated
              column = "",     #open or close price over which indicators are to be calculated
              assetname = "",  #assetName
              by_code = False, #True if subset is called by asset code
              assetcode = ""   #assetcode if subset is done by asset codes
             ):
    asset0 = subset(assetname,by_code,assetcode)
    asset0 = moving_avg(data=asset0,period=15,column=column)
    asset0 = bollinger_bands(data=asset0, periods=15, close_col=column)
    trace0 = go.Scatter(x=asset0.time, y=asset0.open,name='Open', #opening price scatter 
                       line = dict(color = '#03b237'),
                       opacity = 0.8)
    trace1 = go.Scatter(x=asset0.time, y=asset0.close,name='Close', # closing price scatter
                       line = dict(color = '#f20252'),
                       opacity = 0.8)

    trace2 = go.Bar(x=asset0.time, y=asset0.volume, #volume traded bar
                    yaxis = 'y2',
                    name = 'Volume',
                    marker = dict(color = '#ff9933')
                    )
    trace3 = go.Scatter(x=asset0.time, y=asset0.sma15,name='Simple Moving Average'+ column,
                       line = dict(color = '#6b20e5'), # SMA Scatter
                       opacity = 0.6)
    trace4 = go.Scatter(x=asset0.time, y=asset0.UpperBand,name='Upper_BB_Band', 
                       line = dict(color = '#c0c3c6'), # upper Bollinger band
                       opacity = 0.6)
    trace5 = go.Scatter(x=asset0.time, y=asset0.LowerBand,name='Lower_BB_Band',
                       line = dict(color = '#c0c3c6'), # lower bollinger band
                       opacity = 0.6)
    r = max(asset0.volume)*math.log10(max(asset0.open)) # scaling y axis for showing volume
    data = [trace0,trace1,trace2,trace3,trace4,trace5] 
    layout1 = go.Layout(
        title= assetname+assetcode ,
        yaxis=dict(
            title='Stock Price'
        ),
        yaxis2=dict(
            title='Volume',
            range=[0, r],
            overlaying='y',
            side = 'right'
        )
    )
    layout1.update(layout)
    fig = dict(data = data ,layout = layout1)
    py.iplot(fig)


# In[ ]:


#we can either call the function by asset name or code
print("First entry: " , assetNames[0])
make_plot(assetname=assetNames[0], column='open',period=15, layout=layout)


# In[ ]:


#we can either call the function by asset name or code
print("400th entry: " , assetNames[400])
make_plot(assetname=assetNames[400],column='open',period=15, layout=layout)


# In[ ]:


print("1000th entry by code: ", assetCodes[1000])
make_plot(assetcode=assetCodes[1000],by_code= True,column='open',period=15, layout=layout)


# In[ ]:


print("100th entry by code: ", assetCodes[100])
make_plot(assetcode=assetCodes[100],by_code= True,column='open',period=15, layout=layout)


# **Types of Returns**
# *For example we'll look for the first stock in assetNames*
# 1. Returns calculated close-to-close (from the closing time of one trading day to the closing time of anotherc & not adjusted) for 1 day. *returnsClosePrevRaw1*
# 2. Returns calculated open-to-open (from the opening time of one trading day to the opening time of another & not adjusted) for 1 day. *returnsOpenPrevRaw1*
# 3. Returns calculated close-to-close (from the closing time of one trading day to the closing time of anotherc & market adjusted) for 1 day. *returnsClosePrevMktres1*
# 4. Returns calculated open-to-open (from the opening time of one trading day to the opening time of another & market adjusted) for 1 day. *returnsOpenPrevMktres1*
# 5. Returns calculated close-to-close (from the closing time of one trading day to the closing time of anotherc & not adjusted) for 10 days. *returnsClosePrevRaw10*
# 6. Returns calculated open-to-open (from the opening time of one trading day to the opening time of another & not adjusted) for 10 days. *returnsOpenPrevRaw10*
# 7. Returns calculated close-to-close (from the closing time of one trading day to the closing time of anotherc & market adjusted) for 10 days. *returnsClosePrevMktres10*
# 8. Returns calculated open-to-open (from the opening time of one trading day to the opening time of another &  market adjusted) for 10 days. *returnsOpenPrevMktres10*
# 9. Returns calculated open-to-open (from the closing time of one trading day to the closing time of another & market adjusted) for 10 days. *returnsCloseNextMktres10*
# **This is our target variable**
# To predict the stock movement in next 10 days.

# In[ ]:


#function to plot returns(various) 
#bar graphs
def plot_returns(layout: dict,returns_type: str,assetname = "",by_code = False, assetcode = ""):
    asset0 = subset(assetname,by_code,assetcode)
    #asset0['y'] = asset0[returns_type]
    asset0_1 = asset0[asset0[returns_type] >= 0]
    asset0_2 = asset0[asset0[returns_type] < 0]
    trace0 = go.Scatter(x=asset0.time, y=asset0.open,name='Open', #opening price scatter 
                       line = dict(color = '#03b237'),
                       opacity = 0.8)
    trace1 = go.Scatter(x=asset0.time, y=asset0.close,name='Close', # closing price scatter
                       line = dict(color = '#f20252'),
                       opacity = 0.8)
    trace0_1 = {
      'x': asset0_1.time,
      'y': asset0_1[returns_type],
        'name' : "+ve Returns",
        'yaxis':'y2',
      'marker': dict(
            color='#42f453',
            line=dict(
                color='#41f467',
                width=2,
            )),
      'type': 'bar',
      'opacity':0.8
    };
    trace0_2 = {
      'x': asset0_2.time,
      'y': asset0_2[returns_type],
      'name' : "-ve Returns",
        'yaxis':'y2',
      'marker': dict(
            color='#ff3f3f',
            line=dict(
                color='#ff2d2d',
                width=2,
            )),
      'type': 'bar',
      'opacity':0.8
    };
    data = [trace0, trace1, trace0_1, trace0_2];
    layout_bar = {
      'xaxis': {'title': 'Time'},
      'yaxis': {'title': returns_type},
      'barmode': 'relative',
      'title': returns_type,
       'yaxis2': dict(
            title='Return',
            range=[-1, 1],
            overlaying='y',
            side = 'right'
        )
    
    };
    layout_bar.update(layout)
    layout_bar.update({'title': 'Returns of : ' + assetname + " ("+ returns_type + ")"})

    py.iplot({'data': data, 'layout': layout_bar})


# In[ ]:


print( assetNames[101])
#returns_type can be 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
#   'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
#   'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
#   'returnsClosePrevMktres10', 'returnsOpenPrevMktres10','returnsOpenNextMktres10'
plot_returns(assetname=assetNames[101], layout=layout,returns_type='returnsClosePrevRaw1')

