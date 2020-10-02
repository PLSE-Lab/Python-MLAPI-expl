#!/usr/bin/env python
# coding: utf-8

# Thanks vopani for the dataset and Rob and WD for the problem: [Largest Profit or Loss Problem without For loop](https://ryxcommar.com/2019/08/05/a-cool-sql-problem-avoiding-for-loops/)
# 
# **Problem:**
# > You have a table of trading days (with no gaps) and close prices for a stock.
# > Find the highest and lowest profits (or losses) you could have made if you bought the stock at one close price and sold it at another close price, i.e. a total of exactly two transactions.
# > You cannot sell a stock before it has been purchased. 
# >Avoid For loop

# ###  HDFC Bank stock data: Finding the highest profit and loss in 2019

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/nifty50-stock-market-data/HDFCBANK.csv")

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


price = df[['Date','Close']]
price.Date = pd.to_datetime(price.Date, format="%Y-%m-%d")
price=price[(price['Date'] >pd.Timestamp(2019,1,1))]
price.head()


# In[ ]:


plt.plot(price.Date, price.Close)
plt.figure(figsize=(14,7))
plt.gcf().autofmt_xdate() 
plt.show()


# Stock splits happened in 2011-07-14(1:5 ratio) and 2019-09-19(1:2 ratio). So, normalizing the time series.
# 

# In[ ]:


#price.Close = price.Close.mask( price.Date >='2011-07-14', price.Close*5)
price.Close = price.Close.mask( price.Date >='2019-09-19', price.Close*2)

plt.plot(price.Date, price.Close)
plt.figure(figsize=(60,120))


# In[ ]:


#price.sort_values(by='Date', ascending=False, inplace=True)
price.head()


# * Highest Profit that could have been made

# In[ ]:


price['lowest_cumulative_price']=price.Close.cummin()
price['highest_profit']=price.Close-price['lowest_cumulative_price']
price.highest_profit.max()


# * Biggest Loss that could have been made

# In[ ]:


price['highest_cumulative_price']=price.Close.cummax()
price['highest_loss']=price.Close-price['highest_cumulative_price']
price.highest_loss.min()

