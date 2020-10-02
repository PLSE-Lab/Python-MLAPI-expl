#!/usr/bin/env python
# coding: utf-8

# This is an introduction to machine learning, python and pandas. Outline is as follows:
# * Examine the columns and range of the data
# * Visualize some data
# * Brainstorm  some ideas for machine learning as applied to this data
# * Manipulate the data (create new columns, massage data) toward that end.
# *  Apply some machine learning algrorithms and evaulate effectiveness if any

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output

# Try prophet as seen in https://www.kaggle.com/samuelbelko/predicting-prices-of-avocados
from fbprophet import Prophet
# Not sure what this does - show graphs in line?
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# From the above, I have the data for this dataset. Read it in, and do some simple examinations
data = pd.read_csv('../input/Jan20_NDXT.csv')

# First 10 records
data.head()




# In[ ]:


# Info on the columns - For some reason if I do this in the above, I only see output for one of them so 
# I do separately
data.info()

#Desc


# In[ ]:


# Not alot of data, but we are just playing around for now, let's look at max/min and if there are any 
# null values
data.describe()


# Volume is all 0 so might as well drop that.
# I don't know the difference between close and adjusted close so I will drop adjusted close too.
# Volatility might be a predictor so I want to calculate the daily max-min and also the close-open (signed)
# Wondering if certain months are important so also want to add a column which is the month 
# TODO - drop unneeded columns, add new calculated columns

# In[ ]:


data['DailyDif'] = data['NDXT Close'] - data['NDXT Open']
data['DailyRange'] = data['NDXT High'] - data['NDXT Low']
data.drop(columns=['NDXT Volume','NDXT Close','NDXT High','NDXT Low','NDXT Adj Close'], inplace=True)
data.head()


# In[ ]:





# In[ ]:


#I want to convert the date to something Pandas likes
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.head()


# In[ ]:


#Plot theh open price versus time
data.plot(x='Date', y='NDXT Open', kind="line")


# In[ ]:


#Next I want to create a column called month. So I tried this
# data['Month'] = data['Date'].month
# But that didn't work, even though I looked up python datetime and it supports that property
# The error I got said that 'object' did not have the property 'month' so i guess i need some kind of
# cast. So i try
# data['Data'].astype(datetime) but that doesn't work, it says it doesn't recognize datetime
# Finally I use data.info() to dump the columns and their types and i see that Date is of type
# datetime64. But no matter what I did, I couldn't get it ti work. Finally I found some example where
# I see someone refer to a dt property. Now it works - see below
data.info()


# In[ ]:





# In[ ]:





# In[ ]:


# Use dt property to access the actual date time properties
data['Month'] = data['Date'].dt.month
data.head()


# I want to try this Prophet which I saw in the avocado price kernal

# In[ ]:


pdata = data[['Date', 'NDXT Open']].reset_index(drop=True)
pdata = pdata.rename(columns={'Date':'ds', 'NDXT Open':'y'})
pdata.head()


# In[ ]:


m = Prophet()
m.fit(pdata)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[ ]:


fig1 = m.plot(forecast)


# Hey look at my prediction!!! It says the nasdaq is going to keep going up! Buy! Buy! Buy!
# Okay, I just sunk all our savings into Nasdaq. Now is a good time to try the prediction again starting at 2017
# 
# 

# In[ ]:


pdata = pdata[pdata.ds.dt.year < 2017]
pdata.tail()


# In[ ]:


m = Prophet()
m.fit(pdata)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig1 = m.plot(forecast)


# Well this was fun. Not sure what I learned. But it was easy to run Prophet and maybe it would be more interesting to try over a longer date range. But next time I want to go back to my dataframe and try some of the standard ML algorithms
# I want to manipulate the data into the following:
# 
# One row per month (summarize over days):
# Date
# Month (1-12)
# Month closing price
# Gain/loss for the month
# Maximum max-min
# Average max-min
# Same for the previous month
# Same for the previous previous month
# So I guess I need to look at how pandas does grouping!!

# In[ ]:


data.describe(include='all')
#I can see that I have unique dates (good) but not values for every row. So let's look at those.



# In[ ]:


data.loc[data['NDXT Open'].isnull()]
data = data.loc[data['NDXT Open'].isnull()==False]
data.describe(include='all')

