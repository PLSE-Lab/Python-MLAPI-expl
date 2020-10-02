#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import fbprophet
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import plotly.graph_objs as go
import plotly as py
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/btcusd.csv")


# # Convert ds column to datetime format

# In[ ]:


# important: For fbprophet, dataframe column names must be 'ds' and 'y'. Their types must be datetime and float.
df['ds'] = pd.to_datetime(df['ds'])


# # Convert y column to float

# In[ ]:


df['y']=df['y'].astype(float)


# # Get info

# In[ ]:


df.info()


# # Describe dataframe details

# In[ ]:


df.describe()


# # Draw pricing graph

# In[ ]:


# Actually there is no need to draw this graph. Actual data is included the estimation chart.
# plt.plot(df['ds'],df['y'])
# plt.title("BTC Prices in USD")
# plt.ylabel('Price (USD)')
# plt.xlabel('Dates')
# plt.savefig('btc01.png')
# plt.show()


# # Apply forecasting for 4 months ( 123 days)

# In[ ]:


df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15, daily_seasonality=True)
df_prophet.fit(df)

# Forecast for 4 months
fcast_time=123   # 4 months
df_forecast = df_prophet.make_future_dataframe(periods= fcast_time, freq='D')

# Do forecasting
df_forecast = df_prophet.predict(df_forecast)


# # Visualize the forecasting results

# In[ ]:


df_prophet.plot(df_forecast, xlabel = 'Dates', ylabel = 'BTC Price in USD')


# In[ ]:


plt.savefig('btc02.png')
plt.show()


# # Draw plotly interactive graph

# In[ ]:


trace = go.Scatter(
        name = 'Actual price',
       mode = 'markers',
       x = list(df_forecast['ds']),
       y = list(df['y']),
       marker=dict(
              color='#FFBAD2',
              line=dict(width=1)
       )
)


# In[ ]:


trace1 = go.Scatter(
    name = 'trend',
       mode = 'lines',
       x = list(df_forecast['ds']),
       y = list(df_forecast['yhat']),
       marker=dict(
              color='red',
              line=dict(width=1)
       )
)


# In[ ]:


upper_band = go.Scatter(
    name = 'upper band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_upper']),
        line= dict(color='#57b88f'),
        fill = 'tonexty'
)


# In[ ]:


lower_band = go.Scatter(
    name= 'lower band',
        mode = 'lines',
        x = list(df_forecast['ds']),
        y = list(df_forecast['yhat_lower']),
        line= dict(color='#1705ff')
)


# In[ ]:


data = [trace, trace1, lower_band, upper_band]


# In[ ]:


layout = dict(title='Bitcoin Price Estimation Using FbProphet',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=False))


# In[ ]:


figure=dict(data=data,layout=layout)


# In[ ]:


plt.savefig('btc03.png')
py.offline.iplot(figure)

