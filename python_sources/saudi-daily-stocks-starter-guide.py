#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.style.use('ggplot')


# In[ ]:


# load on data set
df_stc = pd.read_csv('/kaggle/input/saudi-daily-stocks-history-test/STC_7010.csv')


# In[ ]:


# look at shape
df_stc.shape


# In[ ]:


df_stc.dtypes


# In[ ]:


# convert date to datetime
df_stc.date = pd.to_datetime(df_stc.date)


# In[ ]:


# plot closing prices
plt.figure(figsize=(8,6))
plt.plot(df_stc.date, df_stc.close)
plt.xlabel("Year")
plt.ylabel('Price (SAR)');
plt.title("STC Stock Price");


# Notice that for STC, the stock was going up initially (with some unexpected downwards spike at 2005). The Saudi stock market had a steep downturn in 2006, which is reflected in the figure. Also, notice the recent coronavirus (COVID-19) outbreak effected the prices negatively. Now, let's deal with the downwards spike. There are two ways to handle it:
# - Clean up this point
# - A better way in my opinion is to use 5-window moving average to handle such fluctuation.
# 
# Generally, in stocks prediction, using moving-average (rolling-average or running-average) can gives more realistic visualization of the data as it cleans up the noise and emphasize the general trends more.

# In[ ]:


# plot moving-average
plt.figure(figsize=(8,6))
plt.plot(df_stc.date, df_stc.close.rolling(window=30).mean())
plt.xlabel("Year")
plt.ylabel('Price (SAR)');
plt.title("STC Stock Price with 30-day Moving-average");


# Notice that having a moving-average with 30-window is significantly smoothening the dataset

# In[ ]:


df_stc.value_traded_SAR = df_stc.value_traded_SAR.str.replace(',','').astype(float)


# In[ ]:


# plot closing prices
plt.figure(figsize=(8,6))
plt.plot(df_stc.date, df_stc.value_traded_SAR)
plt.xlabel("Year")
plt.ylabel('Value Traded (SAR)');
plt.title("STC Stock Value Traded");


# We can see that from the plot, STC was very popular company at the begining. However, after the Saudi stock downturn in 2006, it became not very popular compnay.

# In[ ]:


# plot moving-average
plt.figure(figsize=(8,6))
plt.plot(df_stc.date, df_stc.value_traded_SAR.rolling(window=30).mean())
plt.xlabel("Year")
plt.ylabel('Value Traded (SAR)');
plt.title("STC Stock Value Traded with 30-day Moving-average");


# We can see much smoother trends, allowing us to decide better about the popularity of STC trading value.
# 
# At the end, I hope you find this dataset useful for your analysis.
