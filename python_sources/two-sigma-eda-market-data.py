#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from kaggle.competitions import twosigmanews

from plotly.offline import iplot
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)


# In[ ]:


# retrieve market and news dataframes
env = twosigmanews.make_env()
dfm, dfn = env.get_training_data()


# ### Let's concentrate on the Market Data

# In[ ]:


print('market data shape:', dfm.shape)


# In[ ]:


dfm.head()


# ### Column names are as follows

# In[ ]:


dfm.columns


# ### No. of NAN

# In[ ]:


dfm.isnull().sum()


# ### How many companies are featured?

# In[ ]:


dfm.assetCode.unique().shape[0]


# ### Does companies have an equally wide range (days) of stock prices?

# In[ ]:


days = dfm.groupby('assetCode').size()
days = pd.DataFrame(days, columns=['days']).reset_index().sort_values('days',ascending=False)

# plot barchart
data = [go.Bar(
            x=days.assetCode,
            y=days.days,
            name='No. of Days')]

layout = go.Layout(
    title='No. Companies with Days of Stock Data',
    yaxis=dict(
        title='No. of Days with Data'
    ),
    xaxis=dict(
        title='Company'
    )
)

fig = go.Figure(data=data,layout=layout)

iplot(fig)


# The companies have a very varied time range of stock prices, from 6 years to 1 day!

# In[ ]:


day_count = pd.DataFrame(days['days'].value_counts()).reset_index()
day_count.columns = ['days','count']
day_count = day_count.sort_values('days',ascending=False)

# plot barchart
data = [go.Bar(
            x=day_count['days'],
            y=day_count['count'],
            name='No. of Days')]

layout = go.Layout(
    title='No. Companies with Days of Stock Data',
    yaxis=dict(
        title='No. of Companies'
    ),
    xaxis=dict(
        title='No. of Days with Stock Data'
    )
)

fig = go.Figure(data=data,layout=layout)

iplot(fig)


# However, 500+ companies, or 30+% of all companies have at least 6 years worth of data. It is probably necessary to drop those companies with very low range of trading days, but at least there is a decent amount that have sufficient data.

# ### Let's do some plotting, for the company Apple

# In[ ]:


apple = dfm[dfm['assetCode']=='AAPL.O']

data1 = go.Scatter(
          x=apple.time,
          y=apple['close'],
          name='Price')

data2 = go.Bar(
            x=apple.time,
            y=apple.volume,
            name='Volume',
            yaxis='y2')

data = [data1, data2]

layout = go.Layout(
    title='Closing Price & Volume for AAPL.O',
    yaxis=dict(
        title='Price'
    ),
    yaxis2=dict(
        overlaying='y',
        side='right',
        range=[0, 1500000000], #increase upper range so that the volume bars are short
        showticklabels=False,
        showgrid=False
    )
)

fig = go.Figure(data=data,layout=layout)

iplot(fig)


# There is a huge drop of share price in Jun 2015, but that is because there was a 7:1 stock split. Not sure if this is in the news, if not, stock splits are something that needs to be note as a false positive.
# 
# Note that the volume of trade also increase significantly at the same time, very likely of the simple fact that there are more shares in the market for trading.

# #### and again with a Candlestick chart

# In[ ]:


data = [go.Candlestick(x=apple.time,
                       open=apple.open,
                       high=apple.open, #no data so used another column
                       low=apple.close, #no data so used another column
                       close=apple.close)]

# remove range slider
layout = go.Layout(
    xaxis = dict(
        rangeslider = dict(
            visible = False
        )
    )
)

fig = go.Figure(data=data,layout=layout)
iplot(fig)


# This is not a true candlestick as we were not given the highest and lowest price for each day. But it is still good to know how to plot it using plotly. *Shrugs*

# ### This is the target, ie 10 day forward adjusted opening price returns

# In[ ]:


data = [go.Scatter(
          x=apple.time,
          y=apple['returnsOpenNextMktres10'],
          name='Price')]

iplot(data)


# Just a look at how the target is like...

# ### Which are companies that are more volatile than others?

# In[ ]:


# for the target variable, change negative values into positive
dfm['abs_returnsOpenNextMktres10'] = dfm['returnsOpenNextMktres10'].apply(lambda x: abs(x))
# calculate the mean of absolute returnsOpenNextMktres10 for each company
# and treat that as a proxy for volatility
proxy_beta = dfm.groupby('assetCode', as_index=False)['abs_returnsOpenNextMktres10'].mean()


# plot barchart
data = [go.Bar(
            x=proxy_beta.assetCode,
            y=proxy_beta.abs_returnsOpenNextMktres10)]

layout = go.Layout(
    title='Volatile Companies',
    yaxis=dict(
        title='Proxy Beta'
    ),
    xaxis=dict(
        title='Company'
    )
)

fig = go.Figure(data=data,layout=layout)
iplot(fig)


# The higher the proxy beta, the more volatile the company is. For highly volatile companies, they might be easier to predict as they might respond more drastically to news? Seems like a good hypothesis to test.

# Please come back again for more updates...
