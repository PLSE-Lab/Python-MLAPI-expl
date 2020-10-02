#!/usr/bin/env python
# coding: utf-8

# In this kernel, I'm introducing a really useful python package - Pandas Profiling, using which we can automatically perform basic EDA using just a single line of code, which would otherwise take a lot of time and effort.

# In[ ]:


import pandas_profiling as pp
import pandas as pd 


# In[ ]:


df_chennai_rainfall = pd.read_csv('../input/chennai_reservoir_rainfall.csv', parse_dates=['Date'], index_col='Date')


# In[ ]:


df_report = pp.ProfileReport(df_chennai_rainfall)
df_report


# In[ ]:


df_chennai_reservoirs = pd.read_csv('../input/chennai_reservoir_levels.csv', parse_dates=['Date'], index_col='Date')


# In[ ]:


df_report = pp.ProfileReport(df_chennai_reservoirs)
df_report


# As visible from the above implementation, we get a lot of functionalities with pandas profiling, which might prove to be really helpful for quickly analysing facts or trends in our data.

# Do share this tool with every Machine Learning enthusiast and don't forget to upvote the kernel :)
