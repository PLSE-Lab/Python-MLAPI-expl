#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandas import datetools as dtls
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Analysing only one block

# In[ ]:


block2_hh = pd.read_csv("../input/hhblock_dataset/hhblock_dataset/block_2.csv")
block2_daily  = pd.read_csv("../input/daily_dataset/daily_dataset/block_2.csv")


# In[ ]:


block2_daily.head()


# In[ ]:


block2_daily.info()


# In[ ]:


len(block2_daily['day'].unique())


# In[ ]:


block2_daily = block2_daily.drop(['LCLid'], axis = 1)


# In[ ]:


block2_daily.head()


# In[ ]:


block2_daily = block2_daily[['day', 'energy_sum']]


# In[ ]:


block2_daily.head(12)


# In[ ]:


block2_daily = block2_daily.groupby(['day']).sum().reset_index()


# In[ ]:


block2_daily.head()


# In[ ]:


block2_daily.describe()


# In[ ]:


block2_daily['day'] = pd.to_datetime(block2_daily['day'], format='%Y-%m-%d')


# In[ ]:


block2_daily.info()


# In[ ]:


df = block2_daily


# In[ ]:


df.plot.line(x = 'day', y = 'energy_sum',  figsize=(18,9), linewidth=5, fontsize=20)
plt.show()


# In[ ]:


mon = df['day']


# In[ ]:


temp= pd.DatetimeIndex(mon)


# In[ ]:


month = pd.Series(temp.month)


# In[ ]:


to_be_plotted  = df.drop(['day'], axis = 1)


# In[ ]:


to_be_plotted = to_be_plotted.join(month)


# In[ ]:


sns.barplot(x = 'day', y = 'energy_sum', data = to_be_plotted)


# In[ ]:


to_be_plotted.plot.scatter(x = 'energy_sum', y = 'day', figsize=(16,8), linewidth=5, fontsize=20)
plt.show()


# In[ ]:


# for trend analysis
df['energy_sum'].rolling(5).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.show()


# In[ ]:


# For seasonal variations
df['energy_sum'].diff(periods=30).plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.show()


# In[ ]:


pd.plotting.autocorrelation_plot(df['energy_sum'])
plt.show()


# In[ ]:


pd.plotting.lag_plot(df['energy_sum'])
plt.show()


# In[ ]:


df = df.set_index('day')


# In[ ]:


df.head()


# In[ ]:


# Applying Seasonal ARIMA model to forcast the data 
mod = sm.tsa.SARIMAX(df['energy_sum'], trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
results = mod.fit()
print(results.summary())


# In[ ]:


df['forecast'] = results.predict(start = 735, end= 815, dynamic= True)  
df[['energy_sum', 'forecast']].plot(figsize=(12, 8))
plt.show()


# In[ ]:


def forcasting_future_days(df, no_of_days):
    df_perdict = df.reset_index()
    mon = df_perdict['day']
    mon = mon + pd.DateOffset(days = no_of_days)
    future_dates = mon[-no_of_days -1:]
    df_perdict = df_perdict.set_index('day')
    future = pd.DataFrame(index=future_dates, columns= df_perdict.columns)
    df_perdict = pd.concat([df_perdict, future])
    df_perdict['forecast'] = results.predict(start = 810, end = 810 + no_of_days, dynamic= True)  
    df_perdict[['energy_sum', 'forecast']].iloc[-no_of_days - 12:].plot(figsize=(12, 8))
    plt.show()
    return df_perdict[-no_of_days:]


# In[ ]:


predicted = forcasting_future_days(df,100)


# In[ ]:


predicted


# In[ ]:




