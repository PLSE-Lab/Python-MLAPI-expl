#!/usr/bin/env python
# coding: utf-8

# # Scrape Data from YAHOO or GOOGLE:
# ### Scrape Data from YAHOO or GOOGLE directly with out downloading and uploading to input.
# just a part of code one can add for Stock pridiction.

# In[ ]:


import datetime as dt
from pandas_datareader import data


# In[ ]:


# We would like all available data from 01/01/2000 until Today.
start_date = '2011-01-01'
end_date =  dt.datetime.today()


# In[ ]:


# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_data = data.DataReader('ASHOKLEY.NS', 'yahoo', start_date, end_date)


# In[ ]:


panel_data.head()


# In[ ]:




