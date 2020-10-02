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


# In[2]:


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


# In[3]:


df = pd.read_pickle('../input/all_banks')


# In[4]:


df.head()


# In[5]:


tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']


# What is the max Close price for each bank's stock throughout the time period?

# In[6]:


df.xs(key='Close',axis=1,level='Stock Info').max()


# Create a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock. 
# 

# In[7]:


returns = pd.DataFrame()
for tick in tickers:
        returns[tick+' Return'] = df[tick]['Close'].pct_change()
returns.head()


# We can use pandas pct_change() method on the Close column to create a column representing this return value. Create a for loop that goes and for each Bank Stock Ticker creates this returns column and set's it as a column in the returns DataFrame.

# Create a pairplot using seaborn of the returns dataframe. What stock stands out to you? Can you figure out why?

# In[8]:


sns.pairplot(returns[1:])


# Using this returns DataFrame, figure out on what dates each bank stock had the best and worst single day returns. You should notice that 4 of the banks share the same day for the worst drop, did anything significant happen that day?

# In[9]:


returns.idxmin()


# In[10]:


returns.idxmax()


# Take a look at the standard deviation of the returns, which stock would you classify as the riskiest over the entire time period? Which would you classify as the riskiest for the year 2015?

# In[11]:


returns.std()


# In[13]:


returns.ix['2015-01-01':'2015-12-31'].std()


# Create a distplot using seaborn of the 2015 returns for Morgan Stanley

# In[14]:


sns.distplot(returns.ix['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)


# Create a distplot using seaborn of the 2008 returns for CitiGroup

# In[15]:


sns.distplot(returns.ix['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()


# Create a line plot showing Close price for each bank for the entire index of time. (Hint: Try using a for loop, or use .xs to get a cross section of the data.)

# In[19]:


for tick in tickers:
    df[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()


# Create a heatmap of the correlation between the stocks Close Price

# In[22]:


sns.heatmap(df.xs(key='Close',axis=1,level='Stock Info').corr(),cmap='coolwarm',annot=True)


# Optional: Use seaborn's clustermap to cluster the correlations together:

# In[23]:


sns.clustermap(df.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# In[ ]:




