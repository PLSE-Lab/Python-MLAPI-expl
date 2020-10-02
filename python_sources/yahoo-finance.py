#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install yahoofinance > /dev/null')


# In[ ]:


import yahoofinance as yf
import pandas as pd
import plotly
import plotly.express as px


# In[ ]:


q = yf.HistoricalPrices('CBA.AX', '1900-01-01', '2019-11-15')
df = q.to_dfs()['Historical Prices']


# In[ ]:


df = df.dropna()
df


# In[ ]:


fig = px.line(df.reset_index(), x='Date', y='Adj Close')
fig.show()


# In[ ]:




