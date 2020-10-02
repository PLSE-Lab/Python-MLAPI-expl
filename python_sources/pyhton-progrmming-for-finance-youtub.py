#!/usr/bin/env python
# coding: utf-8

# This none book about one you toub chanal that i decided to learn from it 
# so i prefer to share the code on this course and the link of the course is :
# https://www.youtube.com/playlist?list=PLQVvvaa0QuDcOdF96TBtRtuQksErCEBYZ

# Session 1

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_datareader.data as web 


#read data
style.use('ggplot')

start = dt.datetime(2000,1,1)
end = dt.datetime(2016,12,31)
df = web.DataReader( 'TSLA',"yahoo",start , end)

print(df.head())


# In[ ]:


print(df.tail(6))
df.to_csv('tesla.csv')


# Session 2
# 

# In[ ]:


df = pd.read_csv('tesla.csv'  )
print(df.head())


# In[ ]:


df = pd.read_csv('tesla.csv' , parse_dates=True , index_col=0)
print(df.head())


# visualasation
# 

# In[ ]:


df['Adj Close'].plot()
 


# In[ ]:


df[['Open','High']].plot()
 


# session3
# 

# In[ ]:


df = pd.read_csv('tesla.csv' , parse_dates=True , index_col=0)
print(df.head())


# In[ ]:


df['100ma'] = df['Adj Close'].rolling(window=100).mean()
print(df.tail())


# In[ ]:


df.dropna(inplace=True) # remove all that nan value
df.head()


# In[ ]:


df['100ma'] = df['Adj Close'].rolling(window=100 , min_periods=0).mean()
df.head()


# In[ ]:


df.tail()


# In[ ]:


ax1 = plt.subplot2grid((6,1),(0,0) ,rowspan=5 , colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0) ,rowspan=1 , colspan=1 , sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax1.plot(df.index, df['Open'])
ax2.bar(df.index, df['Volume'])


# Session 4
# 

# In[ ]:


df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()
#pip install https://github.com/matplotlib/mpl_finance/archive/master.zip  in console
from mpl_finance import candlestick_ohlc

import matplotlib.dates  as mdates

print(df_ohlc.head())

df_ohlc.reset_index(inplace=True)

print(df_ohlc.head())
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)


print(df_ohlc.head())

ax1 = plt.subplot2grid((6,1),(0,0) ,rowspan=5 , colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0) ,rowspan=1 , colspan=1 , sharex=ax1)

ax1.xaxis_date()

candlestick_ohlc(ax1,df_ohlc.values ,width=20 ,colorup='g')

ax2.fill_between(df_volume.index .map(mdates.date2num), df_volume.values , 0)


# session 5
