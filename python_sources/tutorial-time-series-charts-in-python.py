#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Time Series Charts in Python
# 
# In this notebook, we will go through plotting time series data in Python, using Matplotlib and Seaborn. Several datasets will be used, and we will touch upon different formats of time series data. This can be used as a quick reference when you want to plot time series charts.

# ## Simplest Plotting: Plot by Year

# For plotting trends by year, we can simply use the **plot() function** putting year as the x-axis. We will use **Suicide Rates Overview dataset** in this example:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sc = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
sc.head()


# We will plot the suicide rate (cases per 100K population) for youth 15-24 years old in Singapore. Firstly with Matplotlib and secondly with Seaborn:

# In[ ]:


sgsc = sc.loc[(sc.country=='Singapore') & (sc.age=='15-24 years'), :]

plt.plot('year','suicides/100k pop', data=sgsc.loc[sgsc.sex=='male',:], c='blue')
plt.plot('year','suicides/100k pop', data=sgsc.loc[sgsc.sex=='female',:], c='red')
plt.legend(('Male','Female'))
plt.show()


# In[ ]:


sns.lineplot(x='year', y='suicides/100k pop', data=sgsc, hue='sex'); # ';' is to avoid extra message before plot


# We will add the following to the plot:
# - A title
# - Showing markers on each data point
# - Enable every labels on the x-axis be shown vertically
# We will also plot a larger chart.

# In[ ]:


plt.figure(figsize=(10,5)) # Figure size
sns.lineplot(x='year', y='suicides/100k pop', data=sgsc, hue='sex', marker='o') # Specify markers with marker argument
plt.title('Suicide Rate in Singapore Aged 15-24') # Title
plt.xticks(sgsc.year.unique(), rotation=90) # All values in x-axis; rotate 90 degrees
plt.show()


# Reference:
# - [List of markers](https://matplotlib.org/api/markers_api.html)
# - [seaborn.lineplot()](https://seaborn.pydata.org/generated/seaborn.lineplot.html)

# ## Monthly and Daily Data
# 
# When we want to plot time series data other than yearly, we need to know how to manipulate time series in Python. In short, we need to (1) **parse the date into Python datetime format** when loading the data; and (2) **set the datetime as the index** of the dataframe. Then plots can be made easily. We will use **ethereum price in Cryptocurrency Historical Prices dataset** as an example:

# In[ ]:


# print(os.listdir("../input/cryptocurrencypricehistory"))
eth = pd.read_csv('../input/cryptocurrencypricehistory/ethereum_price.csv', parse_dates=['Date'])
eth.set_index('Date', drop=True, inplace=True)
eth.sort_index(inplace=True)
eth.head()


# Let's try the simplest plot of close prices:

# In[ ]:


plt.plot(eth.Close);


# We can plot the monthly close price using **.asfreq() method** with 'M' denoting monthly frequency:

# In[ ]:


plt.figure(figsize=(8,6))
plt.plot(eth.asfreq('M').Close, marker='.')
plt.title('Ethereum Monthly Price')
plt.xticks(rotation=90)
plt.show()


# To plot a weekly close price since year 2017, use 'W' in .asfreq() method and subset the the time series:

# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(eth['2017':].asfreq('W').Close, marker='.') # eth['2017':] returns a subset of eth since 2017
plt.title('Ethereum Weekly Price Since 2017')
plt.xticks(rotation=90)
plt.show()


# Please refer to [this part of pandas documentation](http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects) for a list of frequency strings. Chris Albon provides a [simple tutorial](https://chrisalbon.com/python/data_wrangling/pandas_time_series_basics/) on subsetting time series in various methods.

# ## Periodic (e.g. Monthly) Total
# 
# From the raw data, sometimes we want to know the sum of a certain value (e.g. sales) in a certain period (e.g. monthly). Here can use the **.resample() method** in pandas. We will use **Kiva crowdfunding dataset** as an example. We are going the plot the monthly loan amount for Vietnam:

# In[ ]:


kiva = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv', parse_dates=['posted_time'])
kiva_v = kiva.loc[kiva.country=='Vietnam',['posted_time','loan_amount','sector','lender_count']]
kiva_v.set_index('posted_time', inplace=True)
kiva_v.head()


# In[ ]:


plt.figure(figsize=(9,6))
plt.plot(kiva_v.resample('M').sum()['loan_amount'])
plt.title('Kiva Loan Amount in Vietnam')
plt.xticks(rotation=45) # Rotate 45 degrees
plt.show()


# ## Plotting Change
# 
# Sometimes we want to visualize the change of one variable over time. the **.pct_change() function** is very useful.

# In[ ]:


eth.loc[:,'pct_change'] = eth.Close.pct_change()*100
eth.loc['2018':,'pct_change'].plot(kind='bar', color='b')
plt.xticks([])
plt.show()


# Another common visualization is the cumulative change over time. For example, if we invest $1,000 in bitcoin and ethereum, how much will they grow since 2016?

# In[ ]:


# Loading bitcoin data
btc = pd.read_csv('../input/cryptocurrencypricehistory/bitcoin_price.csv', parse_dates=['Date'])
btc.set_index('Date', drop=True, inplace=True)
btc.sort_index(inplace=True)
btc.tail()


# In[ ]:


eth_return = eth['2016-12-31':].Close.pct_change()+1
btc_return = btc['2016-12-31':].Close.pct_change()+1
eth_cum = eth_return.cumprod()
btc_cum = btc_return.cumprod()
plt.figure(figsize=(9,6))
btc_cum.plot(c='blue')
eth_cum.plot(c='cyan')
plt.title('Cumulative Return in Cryptocurrency since 2017')
plt.legend(('Bitcoin','Ethereum'))
plt.yscale('log')
plt.show()


# That's it for now. More plots will be added in the future. Happy plotting!
