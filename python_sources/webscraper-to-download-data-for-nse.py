#!/usr/bin/env python
# coding: utf-8

# # WebScraper to download dataset from NSEPY api

# ![home image](https://www.indianfolk.com/wp-content/uploads/2019/07/1_tkp7o9XLl-RE3hjRZZIC8Q.jpeg)

# In[ ]:


get_ipython().run_line_magic('doctest_mode', '')


# **Note** : you need to Set the internet "toggle" button to 'ON' to successfully execute this kernel.
# 
# #### Steps:
# - go to kernel settings
# - by default internet togglel buttonn is OFF
# - set this button to ON. wait...
# - Then execute as usual.

# In[ ]:


# install the nsepy library (one time only)
get_ipython().system('pip install nsepy')


# #### Read more about NSEPy [here.](https://nsepy.readthedocs.io/en/latest/)

# In[ ]:


# importing all libraries

from nsepy import get_history
from datetime import date
import pandas as pd


# In[ ]:


# setting the year and duration of data collection

from nsepy.derivatives import get_expiry_date
expiry = get_expiry_date(year=2015, month=12)
expiry


# ### Collecting Data for : "TCS,Infosys, NIFTY_IT"

# In[ ]:


tcs = get_history(symbol='TCS',
                   start=date(2015,1,1),
                   end=date(2015,12,31))
infy = get_history(symbol='INFY',
                   start=date(2015,1,1),
                   end=date(2015,12,31))
nifty_it = get_history(symbol="NIFTYIT",
                            start=date(2015,1,1),
                            end=date(2015,12,31),
                            index=True)


# In[ ]:


tcs.head()


# In[ ]:


infy.head()


# In[ ]:


nifty_it.head()


# In[ ]:


#setting index as date
tcs.insert(0, 'Date',  pd.to_datetime(tcs.index,format='%Y-%m-%d') )


# In[ ]:


print(type(tcs.index))
print(type(tcs.Date))
tcs.Date.dt


# In[ ]:


#setting index as date
infy.insert(0, 'Date',  pd.to_datetime(infy.index,format='%Y-%m-%d') )


# In[ ]:


print(type(infy.index))
print(type(infy.Date))
infy.Date.dt


# In[ ]:


#setting index as date
nifty_it.insert(0, 'Date',  pd.to_datetime(nifty_it.index,format='%Y-%m-%d') )


# In[ ]:


print(type(nifty_it.index))
print(type(nifty_it.Date))
nifty_it.Date.dt


# #### Thus, we have created the dataframes for these three stocks. You need to uncomment the below code cell and run it to export these dataframes into 'csv' files.

# In[ ]:


# Uncomment this docstring to export dataframes into csv files.

"""
tcs.to_csv('tcs_stock.csv', encoding='utf-8', index=False)
infy.to_csv('infy_stock.csv', encoding='utf-8', index=False)
nifty_it.to_csv('nifty_it_index.csv', encoding='utf-8', index=False)
"""

