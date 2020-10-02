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

#Other libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__) # requires version >= 1.9.0
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()


# 
# ## Finance crisis up to early 2016.
# 
# 
# **Visualizing the use data optaine from google finance. 
# Second data capstone project for my Udemy Data Science and ML Bootcamp, 
# by Jose Portilla, Udemy.**
# 
# https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp
# 
# **Don't expect much, it is just meant to practice my visualization and pandas skills.
# **
# 
# 
# 

# 
# ## Data
# 
# **We will get stock information for the following banks:**
# 
#     Bank of America BAC
#     CitiGroup C
#     Goldman Sachs GS
#     JPMorgan Chase JPM
#     Morgan Stanley MS
#     Wells Fargo WF
# 
# **I will be using stock data from Jan 1st 2006 to Jan 1st 2016 for each of these banks.**
# 

# In[ ]:


bank_stocks = pd.read_pickle('../input/all_banks') 
bank_stocks.head()


# **Documentation on [Multi-Level](http://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html) Indexing and  [Using .xs](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.xs.html). **
# 
# ** What is the max Close price for each bank's stock throughout the time period?**

# In[ ]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').head(3)


# In[ ]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()


# ** DataFrame returns. This dataframe will contain the returns for each bank's stock. returns are typically defined by:**
# 
# $$r_t = \frac{p_t - p_{t-1}}{p_{t-1}} $$

# In[ ]:


tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
returns = pd.DataFrame()

for x in tickers:
    returns[x + ' Return'] = bank_stocks.xs('Close',axis=1,level=1)[x].pct_change()
returns.head()


# In[ ]:


sns.pairplot(returns[1:])


# ** Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns. You should notice that 4 of the banks share the same day for the worst drop, did anything significant happen that day?**

# In[ ]:


# Worst Drop of returns (4 of them on Inauguration day)
returns.idxmin()


# In[ ]:


# Best Drop of returns (4 of them on Inauguration day)
returns.idxmax()


# ** Standard deviation of the returns, which stock would it classify as the riskiest over the entire time period? Which would it classify as the riskiest for the year 2015?**

# In[ ]:


# Citigroup riskiest
returns.std()


# In[ ]:


returns.loc['2015-01-01':'2015-12-31'].std() 


# In[ ]:


returns.loc['2015-01-01':'2015-12-31'].std() # Very similar risk profiles, but Morgan Stanley or BofA


# ** Create a distplot using seaborn of the 2015/2008 returns for Morgan Stanley **

# In[ ]:


sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)


# In[ ]:


sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)


# In[ ]:


from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
bank_stocks.xs('Close',axis=1,level=1).iplot()


# In[ ]:


# This same plot can be done with a for loop
import matplotlib.pyplot as plt
for tick in tickers:
     bank_stocks[tick]['Close'].plot(label=tick,figsize=(6,2))
plt.legend()


# ## Moving Averages
# 
# Let's analyze the moving averages for these stocks in the year 2008.
# 
# ** Plot the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008**
# 

# In[ ]:



rolling_avg = pd.DataFrame()
rolling_avg['30 Day Avg'] = bank_stocks.loc['2008-01-01':'2008-12-31'].xs('Close',axis=1,level=1)['BAC'].rolling(window=30).mean()

rolling_avg['30 Day Avg'].plot(figsize=(12,6),label='30 Day Avg')
bank_stocks.loc['2008-01-01':'2008-12-31'].xs('Close',axis=1,level=1)['BAC'].plot(figsize=(12,6),label='BofA Close')
plt.legend()


# In[ ]:


sns.heatmap(bank_stocks.xs('Close',axis=1,level=1).corr(),annot=True)


# In[ ]:


sns.clustermap(bank_stocks.xs('Close',axis=1,level=1).corr(),annot=True)


# In[ ]:




