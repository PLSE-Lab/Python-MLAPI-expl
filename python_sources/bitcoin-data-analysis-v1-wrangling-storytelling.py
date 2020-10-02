#!/usr/bin/env python
# coding: utf-8

# # Bitcoin Data Analysis
# #### Oct 2019  |   Work in Progress   |   Jason Su

# ## Introduction
# 
# Bitcoin was the first cryptocurrency and is still the largest cyrptocurrency by market capitalization. Some people think of it as an investment vehicle (similar to commodities in nature) while others think of it as a store of value (similar to gold). Bitcoin was designed to be a decentralized global virtual currency with minimum friction in transactions and a high level of security due to the nature of its underlying blockchain network. Since its inception Bitcoin has been rapidly adopted by enthusiasts and investors worldwide. Now it is traded on multiple online exchanges on the internet by virtually everyone in the world. Consequently, the number of factors affecting the price movements of the Bitcoins is huge and the underlying mechanisms are complex. Every investor wishes to gain a competitive advantage in predicting the price movements of Bitcoins. In this notebook I try to analyze the relationships between historical Bitcoin price movements and other relevant indicators such as the level of Bitcoin adoption, the level of difficulty in Bitcoin mining, etc, with the objective to gain insights into how different factors would affect the prices of Bitcoins and how this knowledge would convert to a strategic advantage in everyday Bitcoin trading. 

# ## Data Wrangling
# 
# First let's obtain and clean our Bitcoin price dataset to get ready for analysis.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # making plots and charts
import requests # getting data through APIs

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Historical Bitcoin Price Data
# 
# The following dataset comes from [here](https://www.kaggle.com/mczielinski/bitcoin-historical-data) (Bitcoin price data at 1-minute intervals from select exchanges during the time period from Jan 2012 to August 2019): 

# In[ ]:


# Import the data from CSV file and save it to a dataframe

bitstamp = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv')


# In[ ]:


# Inspect the first a few rows of the data to see what potential cleaning is needed

bitstamp.head()


# hmmm.... Seems there are a lot of missing values, and the first column is coded in Unix time.

# In[ ]:


# Check the number of rows, columns and the datatypes of each column

bitstamp.info()


# There are 8 columnns and around 4 million rows of data. The datatypes seem to be fine since they are mostly float64, which is suitable for price data.

# In[ ]:


# Quickly check the statistics of all the data in each column to see if they make sense
# Based on my common sense historically the prices of Bitcoins went from 0 to an all 
# time high of around $20,000 per coin

bitstamp.describe()


# The mean, min and max values of each price column here seems to be in line with my impressions.

# [Future work] Add a test of normality

# In[ ]:


# Inspect the shape of the dataset

bitstamp.shape


# In[ ]:


# There are around 1.2 million rows of data with missing values

bitstamp['Open'].value_counts(dropna = False)


# In[ ]:


# Do a quick histogram plot here to see the distribution of prices

bitstamp['Open'].plot('hist')


# In[ ]:


# Do a quick box plot here to see the distribution of prices

bitstamp.boxplot(column=['Open', 'High', 'Low', 'Close'])


# In[ ]:


# Set the index of the dataset to be the time of each observation in YYYY-MM-DD HH-MM-SS

bitstamp.set_index(pd.to_datetime(bitstamp['Timestamp'], unit='s'), inplace=True, drop=True)


# In[ ]:


# Inspect the dataset again

bitstamp.head()


# In[ ]:


# Fill the missing values using the forward fill method. 
# This method is appropriate here since the missing values in the original dataset were caused
# by the fact that there were not trading actions during those time periods, so it is safe to assume
# that the prices remained constant when there were no trading. Forward fill method takes the latest
# price data up to that point and filled it forward in time.

bitstamp.fillna(method = 'ffill', inplace = True)


# In[ ]:


# Inspect again

bitstamp.head()


# In[ ]:


# Also check the latest data. These values seem to make sense as compared to the actual prices in August this year

bitstamp.tail()


# In[ ]:


# Plot a histogram again to check the price distributions

bitstamp['Close'].plot('hist')


# [Future work] plot histogram in log scale too

# In[ ]:


# Save the useful columns from the original dataset into a new and clean dataset called bitstamp_clean

bitstamp_clean = bitstamp.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']]


# In[ ]:


# Inspect the information of the clean dataset

bitstamp_clean.info()


# The below is a chart of the price of bitcoin going from 2012 to 2019. Similar plots can be found at any website which lists the price of bitcoin.

# In[ ]:


# Plot the time series price data

bitstamp_clean.plot(y='Close')


# The above plot only tells part of the story. To make more sense of such a volatile dataset, let's plot the same values again but in log scales to see the trends in more appropriate scales.

# In[ ]:


# Create a log-log plot of the closing prices for the past 7 years

bitstamp_clean.plot(y='Close', logx=True, logy=True)


# Now the trend of the prices seem to be more clear. There is a steady increasing trend over the past 7 years for Bitcoin prices.

# Now let's import an additional dataset "[Bitcoin My Wallet Number of Users](https://www.quandl.com/data/BCHAIN/MWNUS)" which tells us the number of Bitcoin wallets using My Wallet Services on a global scale. This is an indicator of the degree of adoption of Bitcoins worldwide.

# In[ ]:


# Obtain Bitcoin wallet data from Quandl 
# (which is a dataset of number of wallets hosts using My Wallet Service on each day from 2009 to 2019. )

wallet = pd.read_csv('../input/bitcoin-my-wallet-number-of-users/BCHAIN-MWNUS.csv')


# In[ ]:


# Inspect the first 5 rows to see the latest wallet data

wallet.head()


# In[ ]:


# Inspect the last 5 rows to see the oldest data from 2009

wallet.tail()


# Back then there were only 2 wallets!

# In[ ]:


# Convert the date column to datetime format for easier processing later
# Also rename the columns while we are here

wallet['Date'] = pd.to_datetime(wallet['Date'])
wallet.rename(columns = {'Date': 'Date', 'Value': 'Wallets'}, inplace = True)


# In[ ]:


# Group our Bitcoin price data by day so that it could be plotted on the same scale
# against the daily wallet data

bitstamp_clean_day = bitstamp_clean.resample('D').mean()


# In[ ]:


# Create a date column in the bitstamp_clean_day dataframe

bitstamp_clean_day['Date'] = bitstamp_clean_day.index


# In[ ]:


# Inspect the first 5 rows to confirm that the timestamps are indeed grouped by days

bitstamp_clean_day.head()


# In[ ]:


# Join the two dataframes (bitstamp_clean_day and wallet) by matching their dates columns

df = pd.merge(bitstamp_clean_day, wallet, how='inner', on='Date')


# In[ ]:


# Inspect the first a few rows to confirm the data looks good to go

df.head()


# Now we are ready to visualize the relationship between prices and number of wallets.

# In[ ]:


# Plot both daily prices and daily number of wallets for Bitcoin on the same graph

plt.plot(df['Date'], df['Close'], 'r', df['Date'], df['Wallets']/10000, 'b')
plt.yscale('log')
plt.xlabel('Year')
plt.ylabel ('Price and Number of Wallets')
plt.title('Bitcoin Price compared to the Number of Wallets')
plt.legend(labels = ['Price', 'Wallets'])
plt.show()


# From the above plot it seems that there is some kind of correlation between the number of wallets (which implies the degree of adoption of Bitcoins worldwide) and the prices of Bitcoins on a log scale. Therefore, by monitoring the level of increase/decrease of total number of wallets on a global scale, it is possible to predict the overall trend of Bitcoin prices over the next couple of years. Also it is worth noting that the rate of change for both quantities seem to be slowing down, indicating the level of volatility is being more stablized.

# Now let's import another dataset "[Bitcoin Difficulty](https://www.quandl.com/data/BCHAIN/DIFF)" which is a measure of how difficult it is to find a hash below a given target. This is an indicator of the level of difficulty of Bitcoin mining, which in turn implies the level of scarcity of new Bitcoin supply.

# In[ ]:


# Import the Bitcoin difficulty dataset

diff = pd.read_csv('../input/bitcoin-difficulty/BCHAIN-DIFF.csv')


# In[ ]:


# Rename the columns for easier processing
# Also change the data format of the "Date" column while we are here

diff.rename(columns = {'Date': 'Date', 'Value': 'Difficulty'}, inplace = True)
diff['Date'] = pd.to_datetime(diff['Date'])


# In[ ]:


# Inspect the first a few rows of the dataset

diff.head()


# In[ ]:


# Merge these data with Bitcoin price dataframe for comparison later

df2 = pd.merge(bitstamp_clean_day, diff, how='inner', on='Date')


# In[ ]:


# Inspect the first a few rows of df2

df2.head()


# In[ ]:


# Plot both daily prices and level of difficulty for Bitcoin mining on the same graph

plt.plot(df2['Date'], df2['Close'], 'r', df2['Date'], df2['Difficulty']/100000, 'b')
plt.yscale('log')
plt.xlabel('Year')
plt.ylabel ('Price and Level of Difficulty')
plt.title('Bitcoin Price compared to the Level of Difficulty')
plt.legend(labels = ['Price', 'Difficulty'])
plt.show()


# Again there seems to be a certain kind of correlation here between the level of mining difficulty and price increase over a long run. The level of difficulty of Bitcoin mining has been steadily increasing ever since it was first invented. This difficulty mechanism was hard coded into the blockchain network to ensure that Bitcoins can maintain its scarcity (fixed supply) and therefore prevent the inflation issues that we would experience with traditional currencies. Therefore, it makes perfect economic sense that as the level of supply of Bitcoins decreases its prices would go up.

# This notebook is a work in progress. I will keep updating it in the next couple of months. : )
