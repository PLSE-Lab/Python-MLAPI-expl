#!/usr/bin/env python
# coding: utf-8

# # Time Series Analysis (a BITCOIN- USD case study)
# A Time Series is a sequence of well-defined data points measured at consistent time intervals over a period of time and Time series analysis is a statistical technique that deals with time series data, or trend analysis. 
# 
# ### Why is it Important?
# 1. Helpful in financial, business forecasting based on historical trends and patterns.
# 2. IT is used to study cross-correlation/relationship between time series and their dependency on one another.
# 3. It is used in forecasting sales, profit etc.
# 4. The seasonal variation are very useful for businesses and retailers.
# ...
# 
# In this Notebook, I will be anaylzing the Bitcoin/USD trade data sourced from http://www.cryptodatadownload.com/data/northamerican/ which contains daily trade report from BITSTAMP.NET . I will try to understand the dataset looking for possible linear increase/decrease behavior of the series over time,repeating patterns (if there is any).
# Lets begin by importing & preparing our dataset.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

btc_usd=pd.read_csv('../input/Bitstamp_BTCUSD.csv')
btc_usd.head()


# In[ ]:


#lets see what our dataset looks like
print(btc_usd.dtypes)
btc_usd.describe()


# We can observe from the output above that our date is not displayed in the correct datatype.So  we will make adjustments to the "Date" column's datatype and modify our Dataframe's index.

# In[ ]:


btc_usd['Date'] =btc_usd['Date'].astype(str)+ ' 00:00:00'

btc_usd.Date = pd.to_datetime(btc_usd.Date,format='%m/%d/%Y %H:%M:%S').dt.strftime('%d-%m-%Y %H:%M:%S')
btc_usd.Date = btc_usd.Date.astype('datetime64[ns]')
print(btc_usd.dtypes)
btc_usd.head()


# In[ ]:


btc_usd.set_index('Date', inplace=True)
btc_usd.head()


# Now lets pull some records and see if our data works fine. Say we want all records from 1st MAY- 30 in 2019:

# In[ ]:


btc_usd['2019-05-01':'2019-05-31'].head()


# ## Weekly BTC/USD 
# We will run an operation on our data to transform the data to show the weekly BTC Trade volume from 2014 -2019. 

# In[ ]:


weekly_btcusd=btc_usd.resample('W').mean()
weekly_btcusd.fillna(method='ffill',inplace=True) #handle missing records 
print('Weekly BTC/USD')

weekly_btcusd.head()


# Plot a Time Series graph of Avg. Volume per Weeek.

# In[ ]:


weekly_btcusd['Volume BTC'].plot()


# In[ ]:


weekly_btcusd['Datetime']=weekly_btcusd.index

weekly_btcusd = weekly_btcusd.sort_values('Datetime')
weekly_btcusd['Weeks'] = pd.factorize(weekly_btcusd['Datetime'])[0] + 1
mapping = dict(zip(weekly_btcusd['Weeks'], weekly_btcusd['Datetime'].dt.date))

ax = sns.regplot(x='Weeks',y='Volume BTC',data=weekly_btcusd,scatter_kws={'alpha':0.5},fit_reg=False)
labels = pd.Series(ax.get_xticks()).map(mapping).fillna('')



# ## Monthly BTC/USD 
# Lets run the previous operation on our data to transform the data to show the Monthly BTC Trade volume from 2014 -2019. 

# In[ ]:


monthly_btcusd=btc_usd.resample('M').mean()
monthly_btcusd.fillna(method='ffill',inplace=True)
print('Monthly BTC/USD')

monthly_btcusd.head()


# A Graph Showing BT Trade Volumes Avg. per Month for 2014-2019. 

# In[ ]:


monthly_btcusd['Volume BTC'].plot()


# In[ ]:


monthly_btcusd['Datetime']=monthly_btcusd.index

monthly_btcusd = monthly_btcusd.sort_values('Datetime')
monthly_btcusd['Months'] = pd.factorize(monthly_btcusd['Datetime'])[0] + 1
mapping = dict(zip(monthly_btcusd['Months'], monthly_btcusd['Datetime'].dt.date))

ax = sns.regplot(x='Months',y='Volume BTC',data=monthly_btcusd,scatter_kws={'alpha':0.5},fit_reg=False)
labels = pd.Series(ax.get_xticks()).map(mapping).fillna('')


# ## Yearly BTC/USD 
# Finally we wil run one last form of this operation on our data to transform the data to show the Yearly BTC Trade volume from 2014 -2019. 

# In[ ]:


yearly_btcusd=btc_usd.resample('AS-JAN').mean()
yearly_btcusd.fillna(method='ffill',inplace=True)
print('Yearly BTC/USD')

yearly_btcusd.head()


# A Line plot showing a Summary of BTC Trade Volumesfrom 2014-2019

# In[ ]:


yearly_btcusd['Volume BTC'].plot()


# Histogram showing  Daily BTC Volume Distributions for 2015,2016 and 2017

# In[ ]:


#2015
btc_usd['Volume BTC']['2015-01-01':'2015-12-31'].plot.hist(alpha=0.5)
#2016
btc_usd['Volume BTC']['2016-01-01':'2016-12-31'].plot.hist(alpha=0.5)
#2017
btc_usd['Volume BTC']['2017-01-01':'2017-12-31'].plot.hist(alpha=0.5)


# Here, we try to view a summary and compare the Monthly BTC Volume Avg. for 2014-2019.

# In[ ]:


timeseries=pd.DataFrame(index=monthly_btcusd.index)
timeseries['Volume BTC']=monthly_btcusd['Volume BTC']
timeseries['Datetime']=monthly_btcusd.index
timeseries['Datetime']=timeseries['Datetime'].dt.month


sns.set(style="darkgrid")
mapping=timeseries['2014-01-01':'2014-12-31'].plot(kind='line',x='Datetime',y='Volume BTC',color='red',label='2014')
timeseries['2015-01-01':'2015-12-31'].plot(kind='line',x='Datetime',y='Volume BTC',color='green',label='2015',ax=mapping)
timeseries['2016-01-01':'2016-12-31'].plot(kind='line',x='Datetime',y='Volume BTC',color='blue',label='2017',ax=mapping)
timeseries['2018-01-01':'2018-12-31'].plot(kind='line',x='Datetime',y='Volume BTC',color='orange', label='2018', ax=mapping)
timeseries['2019-01-01':'2019-12-31'].plot(kind='line',x='Datetime',y='Volume BTC',color='purple', label='2019', ax=mapping)
mapping.set_xlabel('Months')
mapping.set_ylabel('Volume BTC')
mapping.set_title('BTC/USD Trade 2014 - 2019')
mapping=plt.gcf()
mapping.set_size_inches(10,6)


# In[ ]:


sns.violinplot(btc_usd['Volume BTC'])


# ## The Augmented Dickey-Fuller test
# This test is used to check whether a given time series is stationary or integrated?
#  
# The Augmented Dickey Fuller Test (ADF) is unit root test to see if the statistical properties (such as mean, variance, autocorrelation, etc) of a time series are all constant over time. This test checks for Trends and Seasonalities. It important to check for unit roots because can cause unpredictable results in time series analysis and most tools used in time analysis work with the assumption that statistical properties do not change.

# In[ ]:


from statsmodels.tsa.stattools import adfuller

score = adfuller(btc_usd['Volume BTC'])

print('Augmented Dickey-Fuller Statistic: %f' % score[0])
print('p-value: %f'%score[1])
print('Critical Values:')
for item, point in score[4].items():
    print('Value at %s = %.2f' % (item, point))


# From the Output above, we have a Statistic value of -5.115664. 
# 
# -5.115664 < -3.43 at 1% 
# 
# We can therefore say that the Time series is Stationary. For more information, visit: 
# 
# https://machinelearningmastery.com/time-series-data-stationary-python/
# 
# ### Contact Me:
# Feel free to contact me via samtheo1597@gmail.com, +2348151475929
