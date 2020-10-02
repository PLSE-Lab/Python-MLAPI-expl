#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In this data analysis, I will focus on exploratory data analysis of stock prices.
# 
# Keep in mind, this project is just meant to practice data visualization and pandas skills, It is not meant to be a robust financial analysis or be taken as financial advice.

# I wil focus on bank stocks and see how they progressed throughout the [financial crisis](https://en.wikipedia.org/wiki/Financial_crisis_of_2007%E2%80%932008) all the way to early 2018.

# # Get the Data

# In this section let us learn how to use pandas to directly read data from Google/morningstar finance using pandas!
# 
# Note: You will need to install [pandas-datareader](https://github.com/pydata/pandas-datareader) for this to work! Pandas datareader allows you to [read stock information directly from the internet](http://pandas.pydata.org/pandas-docs/stable/remote_data.html) Use these links for install guidance (pip install pandas-datareader)
# 
# Lets get started by importing all the required packages for this job.!!!!

# In[1]:


from pandas_datareader import data, wb
import pandas as pd
import numpy as np
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
# 
# ** Figure out how to get the stock data from Jan 1st 2006 to April 30th 2018 for each of these banks. **
# ** Use [this documentation page](http://pandas.pydata.org/pandas-docs/stable/remote_data.html) for hints and instructions. Use google/morningstar finance as a source, for example:**
# 
# I am using monringstar for my reference.
# 

# Here the usage of datetime is bit weird as you have to mentioned it two times.. But thats the way it works, so we need to go as is it. :)

# In[ ]:


start = datetime.datetime(2006,1,1)
end = datetime.datetime.now()


# **Now lets get all the financial data from online web site**

# In[ ]:


'''
# Bank of America
BAC = data.DataReader("BAC", 'morningstar', start, end)

# CitiGroup
C = data.DataReader("C", 'morningstar', start, end)

# Goldman Sachs
GS = data.DataReader("GS", 'morningstar', start, end)

# JPMorgan Chase
JPM = data.DataReader("JPM", 'morningstar', start, end)

# Morgan Stanley
MS = data.DataReader("MS", 'morningstar', start, end)

# Wells Fargo
WFC = data.DataReader("WFC", 'morningstar', start, end)

'''


# ** Could also do this for a Panel Object **
# 
# df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'],'morningstar', start, end)

# **Unfortunately the above mentioned code to get a dump of stock price is not working in Kaggle. It could be due to security (firewall setting at Kaggle servers) issue. 
# So I executed the same in my local PC and uploaded the data to Kaggle for further analysis.  **
# 
# ** For detail on how did i get this data(python code), please refer to the description of data set **
# 
# Lets go ahead with our exploratory data analysis. 

# In[2]:


df = pd.read_pickle('../input/all_banks.pickle')


# Lets check what is there in this data set. 

# In[3]:


df.head()


# This data contain daily status of each bank mentioned above. 
# 
# Now lets find out the max Close price for each bank's stock throughout the time period?

# In[4]:


df.xs(key='Close',axis=1,level='Stock Info').max()


# Create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock. returns are typically defined by:
# 
# $$r_t = \frac{p_t - p_{t-1}}{p_{t-1}} = \frac{p_t}{p_{t-1}} - 1$$

# In[5]:


returns = pd.DataFrame()


# We can use pandas pct_change() method on the Close column to create a column representing this return value. Create a for loop that goes and for each Bank Stock Ticker creates this returns column and set's it as a column in the returns DataFrame.

# In[6]:


tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
for tick in tickers:
    returns[tick+' Return'] = df[tick]['Close'].pct_change()
returns.head()


# Create a pairplot using seaborn of the returns dataframe.

# In[7]:


import seaborn as sns
sns.pairplot(returns[1:])


# You can observe that CITI group has differ pattern then other banks. Reason.... checck below link. 
# 
# Background on [Citigroup's Stock Crash available here.](https://en.wikipedia.org/wiki/Citigroup#November_2008.2C_Collapse_.26_US_Government_Intervention_.28part_of_the_Global_Financial_Crisis.29) 
# 
# You'll also see the enormous crash in value if you take a look a the stock price plot (which we do later in the visualizations.)

# Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns.
# 
# You should notice that 4 of the banks share the same day for the worst drop, did anything significant happen that day?

# In[8]:


# Worst Drop (4 of them on Inauguration day)
returns.idxmin()


# ** You should have noticed that Citigroup's largest drop and biggest gain were very close to one another, did anything significant happen in that time frame? **

# [Citigroup had a stock split.](https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=citigroup+stock+2011+may)

# In[9]:


# Best Single Day Gain
# citigroup stock split in May 2011, but also JPM day after inauguration.
returns.idxmax()


# Take a look at the standard deviation of the returns, which stock would you classify as the riskiest over the entire time period? 
# 
# Which would you classify as the riskiest for the year 2018?

# In[10]:


returns.std()


# Create a distplot using seaborn of returns for Morgan Stanley in last one year.

# In[12]:


sns.distplot(returns.loc['2017-05-01':'2018-04-30']['MS Return'],color='green',bins=100)


# Create a distplot using seaborn of the 2008 returns for CitiGroup

# In[13]:


sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)


# **More visualization**

# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()


# Create a line plot showing Close price for each bank for the entire index of time. 
# 
# Observe citi bank line and time when it has gone down. 

# In[15]:


for tick in tickers:
    df[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()


# In[16]:


df.xs(key='Close',axis=1,level='Stock Info').plot(figsize=(12,4))


# ## Moving Averages
# 
# Let's analyze the moving averages for these stocks in the year 2008. 
# 
# ** Plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008**

# In[17]:


plt.figure(figsize=(12,6))
df['BAC']['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
df['BAC']['Close'].loc['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend()


# ** Plot the rolling 30 day average against the Close Price for Citi bank stock for the year 2008**

# In[19]:


plt.figure(figsize=(12,6))
df['C']['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
df['C']['Close'].loc['2008-01-01':'2009-01-01'].plot(label='C CLOSE')
plt.legend()


# In[18]:


# plotly
df.xs(key='Close',axis=1,level='Stock Info').iplot()


# **Seaborn heatmap of the correlation between the stocks Close Price.**

# In[20]:


sns.heatmap(df.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# **Seaborn's clustermap to cluster the correlations together:**

# In[21]:


sns.clustermap(df.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# In[22]:


close_corr = df.xs(key='Close',axis=1,level='Stock Info').corr()
close_corr.iplot(kind='heatmap',colorscale='rdylbu')


# **Candle plot of Bank of America's stock from Jan 1st 2017  to April 30th 2018.**

# In[23]:


df['BAC'][['Open', 'High', 'Low', 'Close']].loc['2017-01-01':'2018-04-30'].iplot(kind='candle')


# **Simple Moving Averages plot of Morgan Stanley for the year 2017.**

# In[24]:


df['MS']['Close'].loc['2017-01-01':'2018-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')


#  **Create a Bollinger Band Plot for Citi bank for the year 2017.**

# In[26]:


df['C']['Close'].loc['2017-01-01':'2018-01-01'].ta_plot(study='boll')


# From above plots its clear that the trend shows the reason for behavior in Nov 2008 and May 2011. It also shows the current trend of citibank and it looks pretty good.

# In[ ]:




