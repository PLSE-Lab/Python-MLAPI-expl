#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fbprophet')
import datetime
import pandas as pd
import numpy as np
import pylab as pl
import datetime
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from matplotlib.collections import LineCollection
from pandas_datareader import data as wb
from sklearn import cluster, covariance, manifold


# In[ ]:


start = '2004-02-01'
end = '2020-07-04'
#edit your company here
tickers = ['BHARTIARTL.NS']

thelen = len(tickers)

price_data = []
for ticker in tickers:
    prices = wb.DataReader(ticker, start = start, end = end, data_source='yahoo')[['Open','Adj Close']]
    price_data.append(prices.assign(ticker=ticker)[['ticker', 'Open', 'Adj Close']])

#names = np.reshape(price_data, (len(price_data), 1))

names = pd.concat(price_data)
names.reset_index()

#pd.set_option('display.max_columns', 500)

open = np.array([q['Open'] for q in price_data]).astype(np.float)
close = np.array([q['Adj Close'] for q in price_data]).astype(np.float)
print(names)
names = names.reset_index()
open = names.rename(columns={'Date':'ds','Open':'y'})
open = open[['ds','y']]
close = names.rename(columns={'Date':'ds','Adj Close':'y'})
close = close[["ds","y"]]
from fbprophet import Prophet
from plotly.graph_objs import *

df = open

def run(df):
  df = df[4000:]
  #df = df[df['ds'].dt.dayofweek < 5]
  m = Prophet()
  m.fit(df)
  future = m.make_future_dataframe(periods=430)
  future.tail()
  forecast = m.predict(future)
  future = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
  future = future.rename(columns={"yhat":"yhat_future","yhat_lower":"yhat_lower_future","yhat_upper":"yhat_upper_future"})
  mixed = pd.merge(df,future,on="ds",how="inner")
  mixed.tail(30)
  avg = (mixed["yhat_future"].iloc[0]+mixed["yhat_lower_future"].iloc[0]+mixed["yhat_upper_future"].iloc[0])/3
  mixed["avg_val"] = ''
  for i in range(0, len(mixed)):
    avg = (mixed["yhat_future"].iloc[i]+mixed["yhat_lower_future"].iloc[i]+mixed["yhat_upper_future"].iloc[i])/3
    mixed['avg_val'].iloc[i] = avg
  mixed = mixed
  import plotly as py

  import plotly.graph_objects as go
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=mixed["ds"], y=mixed["y"],
                      mode='lines',
                      name='Actual Opening Values'))
  fig.add_trace(go.Scatter(x=mixed["ds"], y=mixed["yhat_future"],
                      mode='lines+markers',
                      name='Predicted Opening Values'))
  fig.show()
  fig.add_trace(go.Scatter(x=mixed["ds"], y=mixed["y"],
                      mode='lines',
                      name='Actual Opening Values'))
  fig.add_trace(go.Scatter(x=mixed["ds"], y=mixed["avg_val"],
                      mode='markers',
                      name='Predicted Opening Avg Values '))

  fig.show()
  return mixed


# In[ ]:


a = run(open)
b = run(close)
a = a.rename(columns={"avg_val":"pred_open_avg","y":"actual_open","yhat_upper_future":"yhat_upper_future_open","yhat_future":"yhat_future_open","yhat_lower_future":"yhat_lower_future_open"})
b = b.rename(columns={"avg_val":"pred_close_avg","y":"actual_close","yhat_upper_future":"yhat_upper_future_close","yhat_future":"yhat_future_close","yhat_lower_future":"yhat_lower_future_close"})
c = pd.merge(a,b,on="ds",how="inner")
df = c[["ds","actual_open","pred_open_avg","actual_close","pred_close_avg"]]
df["open_diff"] = df["pred_open_avg"]-df["actual_open"]
df["close_diff"] = df["pred_close_avg"]-df["actual_close"]
df["actual_raise_of_value"] = df["actual_close"] - df["actual_open"]
df["predicted_raise_of_value"] = df["pred_close_avg"] - df["pred_open_avg"]


# In[ ]:


df.tail(10)


# In[ ]:




