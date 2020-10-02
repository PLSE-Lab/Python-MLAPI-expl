#!/usr/bin/env python
# coding: utf-8

# All About Avocadoes 
# 
# My attempt to dissect a data set of one of my favourite foods and experiment with different seaborn plotting methods and fbprophet forecasting library - this will be an ongoing project as I continue to explore and learn as much as I can. 
# 
# 1) Exploratory Data Analysis using **seaborn** 
# 2) Simple Forecasting using **fbprophet** 

# # Part 1: Exploratory Data Analysis

# In[ ]:


#import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#load data
data = pd.read_csv('../input/avocado-prices/avocado.csv', parse_dates=['Date'])


# In[ ]:


#plot average prices of conventional vs. organic avocadoes over time 
sns.set_style('darkgrid')
sns.set_context('notebook')
sns.relplot(x='Date', y='AveragePrice', data=data, kind='line', hue='type', height=6, aspect=2);


# In[ ]:


#drill down deeper into the price and volume deltas between conventional and organic avocadoes across the years
display(sns.catplot(x='year', y='AveragePrice', data=data, col='type', kind='bar'));
display(sns.catplot(x='year', y='Total Volume', col='type', kind='bar', data=data, sharey=False));


# In[ ]:


#transform data to discover seasonality trend
data['month']=data.Date.dt.strftime('%b')
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
data.month = pd.Categorical(values=data.month, categories=months, ordered=True)
seasonal = data.groupby(['month','type'], as_index=False)['AveragePrice'].mean().sort_values('month')


# In[ ]:


data.head()


# In[ ]:


#plot seasonal trend of avocado prices
g =sns.relplot(x='month', y='AveragePrice', data=data, kind='line', row='type', ci=None, height=3, aspect=3, facet_kws={'sharey':False, 'sharex':False});


# In[ ]:


#plot seasonal trend of avocado retail volume
g =sns.relplot(x='month', y='Total Volume', data=data, kind='line', row='type', ci=None, height=3, aspect=3, facet_kws={'sharey':False, 'sharex':False});


# In[ ]:


#transform data for geographical analysis, excluding aggregated regions
df = data.groupby(['region', 'year'], as_index=False)['AveragePrice','Total Volume'].mean()
list_to_exclude = ['TotalUS', 'West', 'SouthCentral','Northeast','Southeast','GreatLakes','Midsouth','Plains']
df1 = df[~df.region.isin(list_to_exclude)].sort_values('AveragePrice', ascending=False)


# In[ ]:


#plot the range of average prices per city
sns.catplot(x='AveragePrice', y='region', data=df1, height=10, aspect=1, kind='box');


# # Part 2: Forecasting
# 
# With reference to the guide in this very helpful [article](https://pbpython.com/prophet-overview.html) as an example to use fbprophet

# In[ ]:


#install library
from fbprophet import Prophet


# In[ ]:


#create data subset of Total US conventional avocado prices as the forecast target 
subset = data[(data.region == 'TotalUS') & (data.type == 'conventional')]
subset = subset[['Date', 'AveragePrice']]
subset.set_index('Date').plot();


# In[ ]:


#transform Average Price column to log value 
subset['AveragePrice'] = np.log(subset.AveragePrice)
subset.set_index('Date').plot();


# In[ ]:


#last but not least, rename columns to adhere to prophet API
subset.columns = ["ds", "y"]
subset.head()


# In[ ]:


#create the first model and fit the data to the dataframe:
m1 = Prophet(changepoint_prior_scale=0.15)
m1.fit(subset)


# In[ ]:


#tell prophet to predict out 1 year
future1 = m1.make_future_dataframe(periods=365*2)


# In[ ]:


#make the forecast
forecast1 = m1.predict(future1)


# In[ ]:


#examine the forecasted values yhat and its lower & upper range
forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[ ]:


#convert back the values to actual avocado prices
np.exp(forecast1[['yhat', 'yhat_lower', 'yhat_upper']].head())


# In[ ]:


#plot a pretty graph 
m1.plot(forecast1);


# In[ ]:


#plot various components of the model too 
m1.plot_components(forecast1);


# Here are some findings:
# * Avocados are expected to continue rising in prices
# * They dip in the start of the year and are most expensive during summer period 
# * There was a price spike in 2017, probably due to high demand and a flat supply (esp for conventional avocadoes, whose volume remained the same as 2016) 
# * Unsurprisingly organic avocados are more expensive than conventional avocados 
# * They're most expensive in more affulent cities like San Francisco and New York 
# 
# Future ideas: 
# * Examine other forecasting predictive analysis 
# * Try other data sets including stocks
