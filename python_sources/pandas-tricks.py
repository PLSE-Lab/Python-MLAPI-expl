#!/usr/bin/env python
# coding: utf-8

# **Pandas Trick **
# 
# Want to filter a DataFrame that doesn't have a name?
# Use query() to avoid creating an intermediate variable!

# In[ ]:


import pandas as pd


# In[ ]:


stocks = pd.read_csv('../input/stocks.csv',parse_dates = ['Date'])


# Goal : Filter this DataFrame to show "Close < 100"

# In[ ]:


stocks.groupby('Symbol').mean()


# In[ ]:


1# Use the Query Method

stocks.groupby('Symbol').mean().query('Close > 100')


# In[ ]:


2# Create an intermediate variable

temp = stocks.groupby('Symbol').mean()
temp[temp.Close > 100]


# In[ ]:




