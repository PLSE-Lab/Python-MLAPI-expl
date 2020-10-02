#!/usr/bin/env python
# coding: utf-8

# # <b>Bitcoin Price Estimation and Usage of FbProphet Library with Plotly</b>

# In this project we will learn some basic properties of fbprophet library and we will make some estimations using bitcoin data.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import fbprophet
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# - Daily bitcoin prices in a period of 2*91 days will be estimated. 
# - In the first 91 days period actually there are prices. We will use the graphs to see the correctness of the estimations.
# - The second half will contain estimated future prices.

# In[ ]:


estimated_days=91


# In[ ]:


names = ['Date','Price(USD)']
df = pd.read_csv('../input/btcusd.csv',names=names)
df = df[1:]     # remove the first row


# In[ ]:


df.head()


# Select the required columns. Actual column names might be different. We need only dates and prices columns.

# In[ ]:


df = df[['Date','Price(USD)']]


# In FbProphet library we must use 'ds'  and 'y'  as column names. So we rename the existing columns.

# In[ ]:



df = df.rename(columns={'Date': 'ds', 'Price(USD)': 'y'})


# In[ ]:


df.head()


# **Check column types**

# In[ ]:


df.info()


# **Change date format to datetime (if not).**

# In[ ]:


df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')


# **Change price values to numeric if it is not.**

# In[ ]:


df['y'] = pd.to_numeric(df['y'],errors='ignore')


# **Select the starting date and make a copy of df as df0, then remove last "estimated_days" from df (we will estimate these values).**

# In[ ]:


df = df[df['ds']>='2014-01-01']
df0=df.copy()
df = df[:-estimated_days]


# In[ ]:


# Sort records by ds if not sorted
# df = df.sort_values(by=['ds'],ascending=True)


# **Prepare Prophet() model and apply fit() method to that dataframe**

# In[ ]:


df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15,yearly_seasonality=True,daily_seasonality=True)
df_prophet.fit(df)


# **I am using here 2 * estimated_days: half of it has actual prices to compare with trend values. The second half are new estimated future values.**

# In[ ]:


df_forecast = df_prophet.make_future_dataframe(periods= estimated_days*2, freq='D')


# **Forecast future prices**

# In[ ]:


df_forecast = df_prophet.predict(df_forecast)


# **Visualize the results using fbprophet's plot_components() and plot() methods**

# In[ ]:


# plot_components() draws 4 graps showing:
#     - trend line
#     - yearly seasonality
#     - weekly seasonality
#     - daily seasonality
df_prophet.plot_components(df_forecast)

# Draw forecast results
df_prophet.plot(df_forecast, xlabel = 'Date', ylabel = 'Bitcoin Price (USD)')

# Combine all graps in the same page
plt.title(f'{estimated_days} daily BTC/USD Estimation')
plt.title('BTC/USD Price')
plt.ylabel('BTC (USD)')
plt.show()


# **Now prepare for plotly graph.**

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
        line=dict(width=3)
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


tracex = go.Scatter(
    name = 'Actual price',
   mode = 'markers',
   x = list(df0['ds']),
   y = list(df0['y']),
   marker=dict(
      color='black',
      line=dict(width=2)
   )
)


# **Append traces into data list. Prepare the layout and figure.**

# In[ ]:


data = [tracex, trace1, lower_band, upper_band, trace]

layout = dict(title='Bitcoin Price Estimation Using FbProphet',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

figure=dict(data=data,layout=layout)


# In[ ]:


plt.savefig('btc02.png')


# **Draw plotly interactive graph.**

# In[ ]:



py.offline.iplot(figure)
# plt.show()


# **Now, lets change the start date of the time series to 2016 an redraw the graph.**

# In[ ]:


df = df0.copy()
df = df[df['ds']>='2016-01-01']
df1=df.copy()
df = df[:-estimated_days]
df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15,yearly_seasonality=True,daily_seasonality=True)
df_prophet.fit(df)
df_forecast = df_prophet.make_future_dataframe(periods= estimated_days*2, freq='D')
df_forecast = df_prophet.predict(df_forecast)
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
trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat']),
    marker=dict(
        color='red',
        line=dict(width=3)
    )
)
upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_upper']),
    line= dict(color='#57b88f'),
    fill = 'tonexty'
)
lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_lower']),
    line= dict(color='#1705ff')
)
tracex = go.Scatter(
    name = 'Actual price',
   mode = 'markers',
   x = list(df1['ds']),
   y = list(df1['y']),
   marker=dict(
      color='black',
      line=dict(width=2)
   )
)
data = [tracex, trace1, lower_band, upper_band, trace]

layout = dict(title='Bitcoin Price Estimation Using FbProphet',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

figure=dict(data=data,layout=layout)
plt.savefig('btc03.png')
py.offline.iplot(figure)


# **And now lets change the starting date to 2017, and redraw the same graph:**

# In[ ]:


df = df1.copy()
df = df[df['ds']>='2017-01-01']
df2=df.copy()
df = df[:-estimated_days]
df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15,yearly_seasonality=True,daily_seasonality=True)
df_prophet.fit(df)
df_forecast = df_prophet.make_future_dataframe(periods= estimated_days*2, freq='D')
df_forecast = df_prophet.predict(df_forecast)
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
trace1 = go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat']),
    marker=dict(
        color='red',
        line=dict(width=3)
    )
)
upper_band = go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_upper']),
    line= dict(color='#57b88f'),
    fill = 'tonexty'
)
lower_band = go.Scatter(
    name= 'lower band',
    mode = 'lines',
    x = list(df_forecast['ds']),
    y = list(df_forecast['yhat_lower']),
    line= dict(color='#1705ff')
)
tracex = go.Scatter(
    name = 'Actual price',
   mode = 'markers',
   x = list(df2['ds']),
   y = list(df2['y']),
   marker=dict(
      color='black',
      line=dict(width=2)
   )
)
data = [tracex, trace1, lower_band, upper_band, trace]

layout = dict(title='Bitcoin Price Estimation Using FbProphet',
             xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))

figure=dict(data=data,layout=layout)
plt.savefig('btc04.png')
py.offline.iplot(figure)


# **CONCLUSION**
# 
# * FbProphet library makes the time series predictions very easily. 
# * However, the selected portion of the time series, directly affects the results.
# * In the above examples, the second and the third graphs look better than the first one.
# * So, having more data does not mean getting better results everytime.
# * In order to get meaningful results, we should check different date scopes.
# 
# 

# In[ ]:




