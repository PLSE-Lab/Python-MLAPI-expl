#!/usr/bin/env python
# coding: utf-8

# Hello! We are Victor, Shawn and Yi Jian from Team Nameless! 
# 
# Our project is on Stocks Data.
# ![](https://i.kym-cdn.com/photos/images/newsfeed/001/499/826/2f0.png)

# # Section 1: Dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.use_inf_as_na = True
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import gmean
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


# # Section 2: Problem Definition
# 
# Modern Portfolio Theory (MPT) is a Nobel Prize-winning economic theory. It eplains how risk-averse investors can construct portfolios to optimize or maximize expected return based on a given level of market risk.
# 
# ## Two main types of financial data:
# Fundamental Data: Data on the financial status of a company 
# One example is Earnings Per Share, which is the ratio of a company's earnings to the total number of shares.
# 
# Technical Data: Data on the price movements of a stock
# One example is Sharpe Ratio, which is a measure of return to risk. 
# 
# ## Our Problem:
# We want to predict a stock's technical data based on its fundamental data. Specifically, we want to predict the next-year's Sharpe Ratio for a stock based on one year's ratio. This should help us construct a better portfolio. 

# # Section 3: Data Preparation

# In[ ]:


price_df = pd.read_csv('../input/nyse/prices.csv')
sec_df = pd.read_csv('../input/nyse/securities.csv')
fund_df = pd.read_csv('../input/nyse/fundamentals.csv')


# ## Price Dataset Preparation

# In[ ]:


price_df.head()


# In[ ]:


price_df.isna().sum()


# No missing data! That's great. Let's do some renaming of columns, and calculate the daily return on each stock.

# In[ ]:


sec_df = sec_df.rename(columns = {'Ticker symbol' : 'symbol','GICS Sector' : 'sector'})
price_df  = price_df.merge(sec_df[['symbol','sector']], on = 'symbol')
price_df['date'] = pd.to_datetime(price_df['date'])
price_df['year'] = price_df['date'].map(lambda x: x.year)
#return here is log-returns
price_df['return'] = np.log(price_df.close / price_df.close.shift(1)) + 1
#delete entries at the start of each stock
price_df['return_valid'] = price_df['symbol'] == price_df['symbol'].shift(1)
price_df = price_df.drop(price_df[price_df['return_valid'] == False].index)
price_df.dropna(how='any', thresh=None, subset=None, inplace = True)

price_df.tail()


# Sharpe Ratio is only valid if the returns are normally distributed. Let's take a look at whether this is the case.

# In[ ]:


sns.distplot(price_df['return'])
price_df['return'].describe()


# Wow, there are a lot of outliers! Let's remove any outliers 5 standard deviations from the mean.

# In[ ]:


std = price_df['return'].std()
mean = price_df['return'].mean()
low = mean-std*5
high = mean+std * 5
fig, ax = plt.subplots(figsize = [25,5])
sns.distplot(np.clip(price_df['return'],low,high), ax = ax)


# Yup, this looks a lot more normal!

# ## Sector & Fundamental Data Preparation

# Let's rename the columns to make it easier to manipulate, then add the sector to the fundamental data.

# In[ ]:


fund_df = fund_df.rename(columns = {'Ticker Symbol' : 'symbol', 'For Year':'year', 'Period Ending': 'date'})
fund_df  = fund_df.merge(sec_df[['symbol','sector']], on = 'symbol')
fund_df


# ## Sharpe Ratio Calculation

# Sharpe Ratio is a measure of a stock's return-to-risk ratio. It is calculated as follows:
# 
# ![](https://a.c-dn.net/c/content/igcom/en_SG/ig-financial-markets/market-news-and-analysis/trading-strategies/2019/01/17/the-sharpe-ratio-explained/_jcr_content/ArticleContent/image.adaptive.620.high.png/1547722043070.png)
# 

# As the risk free rate differs for different years, let's construct a dictionary to hold the rates for each year. We will take the rate as 10-year US Treasury Yields at year-end, from https://www.macrotrends.net/2016/10-year-treasury-bond-rate-yield-chart 

# In[ ]:


risk_free_rate = {2011:0.0198,
                  2012:0.0178, 
                  2013:0.0304, 
                  2014:0.0217, 
                  2015:0.0227, 
                  2016:0.0244,
                  2017:0.0241}


# Let's calculate the yearly return, std dev and sharpe ratio for the next year, and add that to fund_df. 

# In[ ]:


def f(row):
    next_year_data = price_df[(price_df['symbol'] == row['symbol']) & (price_df['year'] == row['year']+1)]['return']
    current_data = price_df[(price_df['symbol'] == row['symbol']) & (price_df['year'] == row['year'])]
    #print(data)
    try: 
        calculateRatios(row,next_year_data,current_data['close'],risk_free_rate[row['year']+1])  
    finally:
        return row

def calculateRatios(row,data,pricedata,rf): 
    row['closing_price'] = pricedata.iloc[-1]
    row['return'] = (np.mean(data) - 1)*len(data)
    row['stdev'] = data.std()*np.sqrt(len(data))
    row['sharpe'] = (row['return'] - rf) / row['stdev']


# In[ ]:


#check that our code outputs correct result

row = 100
f_sharpe = fund_df[row:row+1].apply(f, axis = 1)['sharpe']
print(fund_df[row:row+1])

data = price_df[(price_df['symbol'] == 'AME') & (price_df['year'] == 2013)]['return']
returns = (np.mean(data) - 1)*len(data)
print(returns)
stdev = data.std()*np.sqrt(len(data))
print(stdev)
sharpe = (returns - risk_free_rate[2013]) / stdev

print("Manually Computed: ", sharpe, " Function Value: ", float(f_sharpe))


# In[ ]:


fund_df = fund_df.apply(f, axis = 1)
fund_df


# In[ ]:


fund_df = fund_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
sns.countplot(fund_df['year'])


# In[ ]:


sns.distplot(fund_df['sharpe'])
fund_df['sharpe'].describe()


# In[ ]:


len(fund_df[fund_df['sharpe'] > 1])/len(fund_df)


# In[ ]:


fund_df.loc[:,'P/E'] = fund_df['closing_price']/fund_df['Earnings Per Share']
fund_df.loc[:,'TEV/NI'] = fund_df['Total Liabilities & Equity']/fund_df['Net Income']
fund_df.drop('closing_price', axis = 1)


# Final Checks!

# In[ ]:


price_df.isna().sum()


# In[ ]:


fund_df.isna().sum().sum()


# In[ ]:


sec_df.isna().sum()


# Alright! Our Dataset Preparation is done :) Time to export it to the next Kernel

# In[ ]:


fund_df.to_csv("fund.csv", index=False)
sec_df.to_csv("sec.csv", index=False)
price_df.to_csv("price.csv", index=False)

