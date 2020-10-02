#!/usr/bin/env python
# coding: utf-8

# # Stock Market Crisis Analysis 
# 
# In this data project we will focus on exploratory data analysis of stock prices.
# 
# We'll focus on bank stocks and see how they progressed throughout the [financial crisis](https://en.wikipedia.org/wiki/Financial_crisis_of_2007%E2%80%9308) all the way to early 2016.

# In[ ]:


from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data
# 
# We need to get data using pandas datareader. We will get stock information for the following banks:
# *  Bank of America
# * CitiGroup
# * Goldman Sachs
# * JPMorgan Chase
# * Morgan Stanley
# * Wells Fargo

# In[ ]:


start=datetime.datetime(2006,1,1)
end=datetime.datetime(2016,1,1)


# In[ ]:


BAC = data.DataReader("BAC", "yahoo", start, end)
C = data.DataReader("C", "yahoo", start, end)
GS = data.DataReader("GS", "yahoo", start, end)
JPM = data.DataReader("JPM", "yahoo", start, end)
MS = data.DataReader("MS", "yahoo", start, end)
WFC = data.DataReader("WFC", "yahoo", start, end)


# In[ ]:


df=data.DataReader(['BAC','C','GS','JPM','MS','WFC'],'yahoo',start,end)


# ** Create a list of the ticker symbols (as strings) in alphabetical order. Call this list: tickers**

# In[ ]:


tickers=['BAC','C','GS','JPM','MS','WFC']


# ** Use pd.concat to concatenate the bank dataframes together to a single data frame called bank_stocks. Set the keys argument equal to the tickers list. Also pay attention to what axis you concatenate on.**

# In[ ]:


bank_stocks=pd.concat([BAC,C,GS,JPM,MS,WFC],keys=tickers,axis=1)
bank_stocks.head(5)


# ** Set the column name levels (this is filled out for you):**

# In[ ]:


bank_stocks.columns.names = ['Bank Ticker','Stock Info']


# ** Check the head of the bank_stocks dataframe.**

# In[ ]:


bank_stocks.head(7)

# EDA

Let's explore the data a bit! Before continuing, I encourage you to check out the documentation on [Multi-Level Indexing](http://pandas.pydata.org/pandas-docs/stable/advanced.html) and [Using .xs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.xs.html).
Reference the solutions if you can not figure out how to use .xs(), since that will be a major part of this project.

** What is the max Close price for each bank's stock throughout the time period?**
# In[ ]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()


# ** Create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock. returns are typically defined by:**
# 
# $$r_t = \frac{p_t - p_{t-1}}{p_{t-1}} = \frac{p_t}{p_{t-1}} - 1$$

# In[ ]:


returns=pd.DataFrame()


# ** We can use pandas pct_change() method on the Close column to create a column representing this return value. Create a for loop that goes and for each Bank Stock Ticker creates this returns column and set's it as a column in the returns DataFrame.**

# In[ ]:


for tick in tickers:
    returns[tick+'return']= bank_stocks[tick]['Close'].pct_change()


# In[ ]:


returns.head()


# ** Create a pairplot using seaborn of the returns dataframe. What stock stands out to you? Can you figure out why?**

# In[ ]:


sns.pairplot(returns[1:])


# ** Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns. You should notice that 4 of the banks share the same day for the worst drop, did anything significant happen that day?**

# In[ ]:


returns.idxmin()
#idxmin is used to find the datetime stamp value for the minimum entries in dataset


# ** You should have noticed that Citigroup's largest drop and biggest gain were very close to one another, did anythign significant happen in that time frame? **

# In[ ]:


returns.idxmax()


# ** Take a look at the standard deviation of the returns, which stock would you classify as the riskiest over the entire time period? Which would you classify as the riskiest for the year 2015?**

# In[ ]:


returns.std()


# In[ ]:


returns.loc['2015-01-01':'2016-01-01'].std().plot()


# ** Create a distplot using seaborn of the 2015 returns for Morgan Stanley **

# In[ ]:


sns.distplot(returns.loc['2015-01-01':'2016-01-01']['MSreturn'],color='green',bins=50)


# ** Create a distplot using seaborn of the 2008 returns for CitiGroup **

# In[ ]:


sns.distplot(returns.['2008-01-01':'2009-01-01']['Creturn'],color='red',bins=70)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()


# ** Create a line plot showing Close price for each bank for the entire index of time. (Hint: Try using a for loop, or use [.xs](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.xs.html) to get a cross section of the data.)**

# In[ ]:


for tick in tickers:
    print(bank_stocks[tick]['Close'].plot(label=tick,figsize=(12,4)));
plt.legend()


# In[ ]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()


# In[ ]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()


# ## Moving Averages
# 
# Let's analyze the moving averages for these stocks in the year 2008. 
# 
# ** Plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008**

# In[ ]:


BAC['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(figsize=(12,4))
BAC['Close'].loc['2008-01-01':'2009-01-01'].plot(label='BAC')
plt.legend()


# ** Create a heatmap of the correlation between the stocks Close Price.**

# In[ ]:


sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# ** Optional: Use seaborn's clustermap to cluster the correlations together:**

# In[ ]:


sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# In[ ]:


close_corr = bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr()
close_corr.iplot(kind='heatmap',colorscale='rdylbu')


# ** Use .iplot(kind='candle) to create a candle plot of Bank of America's stock from Jan 1st 2015 to Jan 1st 2016.**

# In[ ]:


BAC[['Open', 'High', 'Low', 'Close']].loc['2015-01-01':'2016-01-01'].iplot(kind='candle')


# ** Use .ta_plot(study='sma') to create a Simple Moving Averages plot of Morgan Stanley for the year 2015.**

# In[ ]:


MS['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')


# **Use .ta_plot(study='boll') to create a Bollinger Band Plot for Bank of America for the year 2015.**

# In[ ]:


BAC['Close'].loc['2015-01-01':'2016-01-01'].ta_plot(study='boll')


# # The End
