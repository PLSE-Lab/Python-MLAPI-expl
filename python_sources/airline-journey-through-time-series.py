#!/usr/bin/env python
# coding: utf-8

# Here we have Airline passanger data form past some years.Here I will throw light on the concepts of time series data.This kernel is work in process and I will be updating the kernel in coming days.

# ## Input

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas import read_csv


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/5) Recurrent Neural Network/international-airline-passengers.csv',skipfooter=5,index_col="Month")


# In[ ]:


df.tail()


# ### Renaming columns

# In[ ]:


df.rename(columns={'International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60': 'Passengers'}, inplace=True)


# ### Dropping missing values if any

# In[ ]:


df.dropna(inplace=True)


# ### Converting index to Datetime

# In[ ]:


df.index


# We can see that the dates are string which need to be converted to datetime.

# In[ ]:


df.index = pd.to_datetime(df.index)


# In[ ]:


df.index


# We have converted the index to Datetime Index.

# ### Simple Moving Average

# In[ ]:


df['6-month-SMA'] = df['Passengers'].rolling(window=6).mean()
df['12-month-SMA'] = df['Passengers'].rolling(window=12).mean()


# In[ ]:


df.plot(figsize=(10,8));


# Some of the weakness of Simple moving average 
# 
# 1.Smaller window will lead to more noise 
# 
# 2.It will always lag by the size of the Window
# 
# 3.It will never reach to full peak or valley of the data due to the averaging 
# 
# 4.Does not really inform about possible future behaviour,all it really does is describe trends in your data
# 
# 5.Extreme historical values can skew your SMA Significantly.

# ### Exponentially Weighted Moving Average (EWMA) 
# 
# It will allow us to reduce the lag effect from SMA and it will put more weight on values that occured more recently by applying more weight to the more recent values.
# 

# In[ ]:


df['EWMA-12'] = df['Passengers'].ewm(span=12).mean()
df[['Passengers','EWMA-12']].plot(figsize=(10,8));


# We can see that the Seasonality is more clear in the end of the plot.This is because we have weighed the values more towards the end.

# ### Error Trend Seasonality

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['Passengers'],model='multiplicative')
result


# #### Seasonal Component

# In[ ]:


result.seasonal.plot(figsize=(10,8))


# #### Trend

# In[ ]:


result.trend.plot(figsize=(10,8))


# #### All Plots Together

# In[ ]:


fig =result.plot()


# So we have ploted trend,Seasonality and the Resudal component of the Curve.

# #### ARIMA Model 

# In[ ]:


time_series = df['Passengers']


# In[ ]:


type(time_series)


# In[ ]:


time_series.rolling(12).mean().plot(label='12 Month Rolling Mean',figsize=(10,8))
time_series.rolling(12).std().plot(label='12 STD Mean',figsize=(10,8))
time_series.plot()
plt.legend();


# We can see that there is an upward trend shown by the Rolling average curve,This is also quite obvious to our eyes.From the standard deviation plot we will be able to see the years with major deviation or the outlier years>

# In[ ]:




