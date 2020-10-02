#!/usr/bin/env python
# coding: utf-8

# In this data project, I will focus on exploratory data analysis of stock prices for the big automakers in the US:
#     General Motors (GM)
#     Ford (F)
#     Fiat Chrysler (FCAU)
#     Tesla (TSLA)
# from 2015-1-1 to 2018-1-1.

# In[1]:


from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


start = datetime.datetime(2015,1,1)
end = datetime.datetime(2018,1,1)


# In[38]:


#GM
GM = data.DataReader('GM','iex',start,end)

#Ford
F = data.DataReader('F','iex',start,end)

#Fiat Chrysler
FCAU = data.DataReader('FCAU','iex',start,end)

#Tesla
TSLA = data.DataReader('TSLA','iex',start,end)


# In[39]:


#FCAU


# In[40]:


tickers = ["GM","F","FCAU","TSLA"]


# In[41]:


auto_stocks = pd.concat([GM,F,FCAU,TSLA],axis=1,keys=tickers)


# In[44]:


#auto_stocks.head()


# In[43]:


auto_stocks.columns.names = ['Ticker','Stock Info']


# In[45]:


auto_stocks.head()


# In[46]:


auto_stocks.xs(key='close',axis=1,level='Stock Info').max()


# In[47]:


returns = pd.DataFrame()


# In[49]:


for tick in tickers:
    returns[tick+' Return'] = auto_stocks[tick]['close'].pct_change()


# In[50]:


returns.head()


# In[51]:


sns.pairplot(returns[1:])


# In[52]:


#Date of the minimum return for each stock
returns.idxmin()


# In[53]:


#Date of the maximum return for each stock
returns.idxmax()


# In[61]:


#Standard deviation for each stock
returns.std()


# The highest risk stock appears to be FCAU while the lowest is Ford. Lets look at the
# standard deviation for just 2016.

# In[60]:


returns.loc['2016-01-01':'2016-12-31'].std()


# FCAU is still the considered the highest risk and is actually higher in 2016 than
# it was in from the overall data. GM is the lowest however Ford and GM are extremely
# close.

# In[77]:


sns.distplot(returns.loc['2016-01-01':'2016-12-31']['F Return'],color='green',bins=50)


# In[75]:


sns.distplot(returns.loc['2016-01-01':'2016-12-31']['FCAU Return'],color='red',bins=50)


# In[71]:


import plotly
import cufflinks as cf
cf.go_offline()


# In[78]:


auto_stocks.xs(key='close',axis=1,level='Stock Info').iplot()


# Looking at the above graph, you can see where the stocks follow similar trends. All stocks dropped in early 2016, with TSLA
# being the most obvious. In 2017, FCAU surpassed Ford in price.

# In[87]:


plt.figure(figsize=(12,4))
TSLA['close'].loc['2016-01-01':'2017-01-01'].rolling(window=30).mean().plot(label='30 day moving avg')
TSLA['close'].loc['2016-01-01':'2017-01-01'].plot(label='BAC Close')
plt.legend()


# In[90]:


sns.heatmap(auto_stocks.xs(key='close',axis=1,level='Stock Info').corr(),annot=True)


# The highest correlation is between TSLA and GM. There may also be correlations between FCAU and GM as well as FCAU and Ford.

# In[91]:


sns.clustermap(auto_stocks.xs(key='close',axis=1,level='Stock Info').corr(),annot=True)


# In[94]:


tsla16 = TSLA[['open','high','low','close']].loc['2016-01-01':'2017-01-01']
tsla16.iplot(kind='candle')


# In[95]:


TSLA['close'].loc['2016-01-01':'2017-01-01'].ta_plot(study='sma',periods=[13,21,55])


# In[96]:


TSLA['close'].loc['2016-01-01':'2017-01-01'].ta_plot(study='boll')


# In[ ]:




