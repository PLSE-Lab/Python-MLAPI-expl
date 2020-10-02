#!/usr/bin/env python
# coding: utf-8

# This notebook is a quick exploration of the Global Terrorism Database in relation to DJIA stock prices. I quickly realized that there isn't enough stock price data in this data set to overlap some of the major US terrorism events that I expected would show correlation to stock price. So I'm publishing this as just the start of a kernel and perhaps come back to it if a more complete set of stock price data is uploaded to Kaggle.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output

print('Data sets imported:')
print(check_output(["ls", "../input"]).decode("utf8"))


# First, the Global Terrorism Database data set is imported and cleaned up.

# In[ ]:


attacks = pd.read_csv('../input/gtd/globalterrorismdb_0617dist.csv',
                      encoding='ISO-8859-1',
                      usecols=[0, 1, 2, 3, 7, 8, 13, 14, 98, 101])
attacks.rename(columns={'eventid':'event_ID',
                        'iyear':'year',
                        'imonth':'month',
                        'iday':'day',
                        'country':'country_code',
                        'country_txt':'country',
                        'nkill':'number_killed',
                        'nwound':'number_wounded',},inplace=True)
attacks['number_killed'] = attacks['number_killed'].fillna(0).astype(int)
attacks['number_wounded'] = attacks['number_wounded'].fillna(0).astype(int)
attacks.loc[attacks.day==0,'day'] = 1
attacks.loc[attacks.month==0,'month'] = 1
attacks['date'] = pd.to_datetime(attacks[['day','month','year']])
attacks = attacks.drop_duplicates(['date', 'latitude', 'longitude', 'number_killed'])
attacks.set_index('date', inplace=True) #change the dataframe index to a date
attacks_usa = attacks[(attacks.country == 'United States')] #reduce to just USA rows
attacks_usa = attacks_usa[['number_killed','number_wounded']] #reduce the number of columns
attacks_usa.head(5)


# Next, a similar import and cleanse approach was taken on the DJIA stock data as well.

# In[ ]:


stocks = pd.read_csv('../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv',
                      encoding='ISO-8859-1')
stocks.columns = [x.lower() for x in stocks.columns]
stocks['date'] = pd.to_datetime(stocks['date'])
stocks.set_index('date', inplace=True) #change the dataframe index to a date
stocks = stocks.resample('D').mean()
stocks = stocks[['close']] #reduce the number of columns
stocks.dropna(axis=0,how='any',inplace=True)
stocks.head(5)


# Calculate the difference in stock by day.

# In[ ]:


stocks['close_lagged'] = stocks.close.shift(1)
stocks['close_diff'] = stocks['close'] - stocks['close_lagged']
stocks = stocks[['close','close_diff']]
stocks.dropna(axis=0,how='any',inplace=True)
stocks.head(5)


# The graph below shows the number killed in US terrorist attacks by day compared to the DJIA stock price close by day. I then leveled both data sets to 0 to 1 so the scale would be consistent for visualizing.

# In[ ]:


plt.figure(figsize=(12, 5))
plt.plot(attacks_usa.number_killed/attacks_usa.number_killed.max())
plt.plot(stocks.close/stocks.close.max())
plt.plot(stocks.close_diff/stocks.close_diff.max())
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
plt.show()


# I had planned to explore ARIMA or Pearson Correlation to compare the two time series data sets but instead I stopped working on this because of the lack of stock price data.
# 
# For fun, I also calculated some z scores on the chart below.

# In[ ]:


cols = list(attacks_usa.columns)
for col in cols:
    col_zscore = col + '_zscore'
    attacks_usa[col_zscore] = (attacks_usa[col] - attacks_usa[col].mean())/attacks_usa[col].std(ddof=0)

cols = list(stocks.columns)
for col in cols:
    col_zscore = col + '_zscore'
    stocks[col_zscore] = (stocks[col] - stocks[col].mean())/stocks[col].std(ddof=0)

plt.figure(figsize=(12, 5))
plt.plot(attacks_usa['number_killed_zscore'])
plt.plot(stocks['close_zscore'])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
plt.show()

