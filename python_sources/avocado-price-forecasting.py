#!/usr/bin/env python
# coding: utf-8

# # Forecasting Average Avocado Price
# 

# In[ ]:


import pandas as pd
from fbprophet import Prophet
import os


# I am going to be using the [Prophet Library](https://facebook.github.io/prophet/docs/quick_start.html) released by Facebook

# In[ ]:


os.listdir('../input/avocado-prices')


# Processing the data in the .csv using Pandas

# In[ ]:


df =  pd.read_csv(r'../input/avocado-prices/avocado.csv', error_bad_lines = False, encoding='latin-1')


# There is a lot of extraneous information here that I don't need to do basic forecasting

# In[ ]:


df.head()


# I am dropping everything besides the Date and Average Price

# In[ ]:


df = df.drop(['Unnamed: 0','Total Volume','4046','4225', '4770', 'Total Bags','Small Bags','Large Bags','XLarge Bags','type','year','region'], axis=1)


# In[ ]:


df.head()


# The Prophet library requires two columns, ds and y. <br/>
# The ds column happens to already be in DateTime format so I don't have to do any converting

# In[ ]:


df.columns = ['ds','y']


# In[ ]:


p = Prophet()
p.fit(df)


# Forecasting one year into the future

# In[ ]:


future = p.make_future_dataframe(periods = 365, include_history = True)
forecast = p.predict(future)


# In[ ]:


figure = p.plot(forecast, xlabel='Date', ylabel='Average Price ($)')


# In[ ]:


figure2 = p.plot_components(forecast)

