#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 list |grep pandas')
from pandas_datareader import data, wb
import numpy as np
import pandas as pd 
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')

start = datetime.datetime(2006,1,1)
end = datetime.datetime(2016,1,1)
BAC = data.DataReader('BAC',"yahoo",start,end,)
C = data.DataReader('c',"yahoo",start,end)
GS = data.DataReader('GS',"yahoo",start,end)
JPM = data.DataReader('JPM',"yahoo",start,end)
MS = data.DataReader('MS',"yahoo",start,end)
WFC = data.DataReader('WFC',"yahoo",start,end)


# In[ ]:


tickers = ['BAC','C','GS','JPM','MS','WFC']


# In[ ]:


Banks_Stock = pd.concat([BAC,C,GS,JPM,MS,WFC],axis = 1,keys =tickers )
Banks_Stock.head()


# In[ ]:


Banks_Stock.columns.names = ['bank ticker','stock info']
Banks_Stock.head()


# In[ ]:


#max close price for each bank throughout the time period
Banks_Stock.xs(key='Close',axis=1,level='stock info').max()


# In[ ]:


#create a new empty DataFrame called returns.this dataframe will contain the returns for each banks stock
returns = pd.DataFrame()


# In[ ]:


for tick in tickers:
    returns[tick+'return']=Banks_Stock[tick]['Close'].pct_change()
returns.head()


# In[ ]:


import seaborn as sns
sns.pairplot(returns[1:])


# In[ ]:


#using return dataframe, figure out on what dates each bank stock had the best and worst single day returns?
returns.head()


# In[ ]:


#best single day for each bank
returns.idxmax()


# In[ ]:


#worst single day for each bank
returns.idxmin()


# In[ ]:


#standard deviation of the stocks ?
returns.std()


# In[ ]:


#standard deviation of the stocks for 2015 ?
returns.ix['2015-01-01':'2015-12-31'].std()

