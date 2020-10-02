#!/usr/bin/env python
# coding: utf-8

# Simple Qualitative analysis of feasibility of arbitrage trade at Indian market

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import datetime
tick_data = pd.read_csv('../input/log_raw.csv')
# Lets convert timestamp column to timestamp
tick_data.timestamp = pd.to_datetime(tick_data.timestamp)


# In[ ]:


# filter-out pre-market data. 03:45 UTC means 9:15 IST
# Contains too much sprade in market depth
tick_data = tick_data.loc[tick_data.timestamp > '2017-06-19 03:45:00.000']


# In[ ]:


# lets pick ACC Cement. NSE is 1510401. BSE is 136247044
acc_nse = tick_data[tick_data.instrument_token == 5633]
acc_bse = tick_data[tick_data.instrument_token == 128104964]


# In[ ]:


# lets see how frequent is bse/nse feed
print(acc_bse.timestamp.diff().fillna(0).mean())
print(acc_bse.timestamp.diff().fillna(0).std())
print(acc_nse.timestamp.diff().fillna(0).mean())
print(acc_nse.timestamp.diff().fillna(0).std())


# In[ ]:


# all records are of single day now. So lets get ged rid of it and keep only time
acc_time = acc_bse['timestamp'].apply(lambda d: d.time())
acc_bse['timestamp'] = acc_time
acc_time = acc_nse['timestamp'].apply(lambda d: d.time())
acc_nse['timestamp'] = acc_time


# In[ ]:


# Now we can make the timestamp column an unique index column for BSE and NSE
acc_bse = acc_bse.set_index(['timestamp'])
# drop the index name. Cause I dont like it
del acc_bse.index.name
acc_nse = acc_nse.set_index(['timestamp'])
# drop the index name. Cause I dont like it
del acc_nse.index.name


# In[ ]:


import matplotlib.pyplot as plt 
from pylab import rcParams
rcParams['figure.figsize'] = 25, 10
fig, ax = plt.subplots()
ax.plot_date(acc_bse.index, acc_bse.last_price,'v-')
ax.plot_date(acc_nse.index, acc_nse.last_price, 'v-')
plt.show()


# Here is the thing to note. It looks like lot of difference in price existed across market. For intraday trade we can exploit arbitrage opportunity. **We need to initiate a position (buy at the exchange, where the value is lower. And sell at the exchange where the value is higher) when the price difference is maximum. And cover the position (sell the same where you bought and buy where you sold) when the price difference is minimum i.e. ideally zero.**
# It seems such opportunities exists. The threshold needs to be fixed as per brokerage and other charges. Will do everything quantitatively later.
