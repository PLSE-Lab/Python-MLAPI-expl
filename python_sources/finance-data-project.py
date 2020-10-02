#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Finance Data Project 
# 
# This is my second data capstone project for my Udemy Data Science and ML Bootcamp. In this data project I will focus on exploratory data analysis of stock prices. Keep in mind, this project is just meant to practice my visualization and pandas skills, it is not meant to be a robust financial analysis or be taken as financial advice.
# ____
# I'll be focusing on bank stocks and see how they progressed throughout the [financial crisis](https://en.wikipedia.org/wiki/Financial_crisis_of_2007%E2%80%9308) all the way to early 2016.

# 
# In this project I will be using pandas to directly read data from yahoo finance using pandas!
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data
# 
# We will get stock information for the following banks:
# *  Bank of America
# * CitiGroup
# * Goldman Sachs
# * JPMorgan Chase
# * Morgan Stanley
# * Wells Fargo
# 
# I will be using stock data from Jan 1st 2006 to Jan 1st 2016 for each of these banks. 

# In[ ]:


bank_stocks = pd.read_pickle('../input/all_banks')


# In[ ]:


tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']


# In[ ]:


bank_stocks.head()


# # EDA
# 
# Let's explore the data a bit and try to answer some basic questions
# 
# ** What is the max Close price for each bank's stock throughout the time period?**

# In[ ]:


bank_stocks.xs('Close', level=1, axis=1).max()


# ** I'm going to create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock. returns are typically defined by:**
# 
# $$r_t = \frac{p_t - p_{t-1}}{p_{t-1}} = \frac{p_t}{p_{t-1}} - 1$$

# In[ ]:


returns = pd.DataFrame()


# ** We can use pandas pct_change() method on the Close column to create a column representing this return value. **

# In[ ]:


for x in tickers:
    returns[x + ' Return'] = bank_stocks.xs('Close',axis=1,level=1)[x].pct_change()
returns.head()


# In[ ]:


import seaborn as sns
sns.set_style('whitegrid')


# In[ ]:


sns.pairplot(returns[1:])


# In the above figure Citigroup's stock clearly stands out.This behavior was due their stock crashing during the financial crisis.
# A background on [Citigroup's Stock Crash is available here.](https://en.wikipedia.org/wiki/Citigroup#November_2008.2C_Collapse_.26_US_Government_Intervention_.28part_of_the_Global_Financial_Crisis.29) 
# 
# You'll also see the enormous crash in value if you take a look at the stock price plot (which we do later in the visualizations.)

# ** Using this returns DataFrame, lets try to figure out on what dates each bank stock had the best and worst single day returns. **

# In[ ]:


returns.idxmin()


# You can notice that 4 of the banks share the same day for the worst drop. This occured on Inauguration Day when Barack Obama took office.

# In[ ]:


returns.idxmax()


# You should have noticed that Citigroup's largest drop and biggest gain were very close to one another. This was because [Citigroup had a stock split](https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=citigroup+stock+2011+may) in May 2011. Even JP Morgan had a stock split, one day after Inauguration day.

# In[ ]:


returns.std() 


# Take a look at the standard deviation of the returns, clearly Citigroup is the riskiest due to greater deviation. However after the Financial Crisis you can see that all banks had a very similar risk profiles.

# In[ ]:


returns.loc['2015-01-01':'2015-12-31'].std()


# In[ ]:


sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'],bins=100,color='green')


# In[ ]:


sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'],bins=100,color='red')


# ____
# # More Visualization

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
from plotly import __version__


# In[ ]:


from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[ ]:


init_notebook_mode(connected=True)


# In[ ]:


bank_stocks.xs('Close',axis=1,level=1).plot(figsize=(12,5))


# In the above figure you can clearly see the enormous crash in value of Citigroup's stock.

# ## Moving Averages
# 
# Let's analyze the moving averages for these stocks in the year 2008. 
# 

# In[ ]:


bank_stocks.loc['2008-01-01':'2008-12-31'].xs('Close',axis=1,level=1)['BAC'].plot(figsize=(12,6),label='BofA Close')
rolling_avg = pd.DataFrame()
rolling_avg['30 Day Avg'] = bank_stocks.loc['2008-01-01':'2008-12-31'].xs('Close',axis=1,level=1)['BAC'].rolling(window=30).mean()
rolling_avg['30 Day Avg'].plot(figsize=(12,6),label='30 Day Avg')
plt.legend()


# In[ ]:


sns.heatmap(bank_stocks.xs('Close',axis=1,level=1).corr(),annot=True,cmap='Blues')


# In[ ]:


sns.clustermap(bank_stocks.xs('Close',axis=1,level=1).corr(),annot=True,cmap='Blues')


# That's all for now. More to come soon...
