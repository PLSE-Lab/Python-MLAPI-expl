#!/usr/bin/env python
# coding: utf-8

# # Official Gold Reserves Per Country Since 1950
# This dataset is taken from the [International Monetary Fund's data site](https://data.imf.org/).
# 
# This dashboard allows you to do the following: 
# 
# - Check the top fifteen countries that have the most gold reserves in any quarter, since 1950
# - Find out which are the top countries that added/reduced their gold reserves in any period of two quarters
# - Plot the quarterly reserves for any country(s) you want, during any period you want to compare how reserves changed in time

# In[ ]:


from IPython.display import IFrame
IFrame(src='https://www.dashboardom.com/gold-reserves', width='100%', height=600)


# ## The Dataset
# It is a simple timeseries showing gold reserves in ounces per country per quarter.

# In[ ]:


import pandas as pd
gold = pd.read_csv('../input/gold_quarterly_reserves_ounces.csv')
gold['Time Period'] = [pd.Period(p) for p in gold['Time Period']]
gold.head()


# In[ ]:


gold.dtypes


# ## Getting the data
# This is a subset of the IMF's International Financial Statistics dataset, and to generate it you first need to download it and filter as follows:

# In[ ]:


# download URL: https://data.imf.org/?sk=388DFA60-1D26-4ADE-B505-A05A558D9A42&sId=1479329132316
# International Financial Statistics dataset
# ifs = pd.read_csv('path/to/file.csv')
# metric_name = 'International Reserves, Official Reserve Assets, Gold (Including Gold Deposits and, If Appropriate, Gold Swapped), Volume in Millions of Fine Troy Ounces , Fine Troy Ounces'
# gold_quarterly_oz = ifs[(ifs['Indicator Name']==gold_res_oz) & (ifs['Time Period'].astype('str').str.contains('Q'))]


# ## The Dashboard
# The code to create the dashboard can be found here: https://github.com/eliasdabbas/gold-reserves
# and the dashboard itself can also be accessed here: https://www.dashboardom.com/gold-reserves
