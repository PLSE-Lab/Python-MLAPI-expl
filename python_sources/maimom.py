#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Get historical stock
df=pd.read_csv('../input/historical_stock_prices.csv')


# In[ ]:


# List of column names:
print(list(df.columns.values))


# In[ ]:


# number of different stocks
print('Number of different stocks: ', len(list(set(df.ticker.unique()))))


# In[ ]:




