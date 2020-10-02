#!/usr/bin/env python
# coding: utf-8

# The [400+ crypto currency pairs at 1-minute resolution](https://www.kaggle.com/tencars/392-crypto-currency-pairs-at-minute-resolution) dataset is updated once per month. However in the fast paced world of crypto currencies this comes close to eternity.
# So instead of waiting weeks for the next update to backtest your trading strategy with the latest bitcoin price data you can fill the gaps yourself.
# In the following it is outlined how to do so. We will start by installing a Python library that allows us to interact with the [Bitfinex API](https://bitfinex.com). If you want more information about the Bitfinex Python library that is used here, have a look at the [GitHub repository](https://github.com/akcarsten/bitfinex_api).

# In[ ]:


# Install the bitfinex_api package
get_ipython().system('pip install bitfinex-tencars')


# Next we import all libraries required for acquiring the data.

# In[ ]:


import bitfinex
import pandas as pd
import numpy as np
import datetime
import time
import os

import matplotlib.pyplot as plt


# Retrieving data from the Bitfinex API is straight forward. However, there is one detail we have to be aware of:
# The API will only return 1000 data points in one query call. 
# This means, if we want Bitcoin data at 1-minute resolution for the last 6 months we will only get data for 1000 minutes instead. To get around this limitation we simply split the larger query into many smaller queries that stay within the 1000 data point limit. Now this comes with another caveat: we are only allowed to make a certain amount of calls to the API otherwise we get blocked. The solution to this is to wait for 1 to 2 seconds after each call and then continue.
# The function below takes care of all these details. It allows us to specify from when to where, at what resolution and for which crypto currency pair we want to retrieve data.

# In[ ]:


# Create a function to fetch the data
def fetch_data(start=1364767200000, stop=1545346740000, symbol='btcusd', interval='1m', tick_limit=1000, step=60000000):
    # Create api instance
    api_v2 = bitfinex.bitfinex_v2.api_v2()

    data = []
    start = start - step
    while start < stop:

        start = start + step
        end = start + step
        res = api_v2.candles(symbol=symbol, interval=interval, limit=tick_limit, start=start, end=end)
        data.extend(res)
        print('Retrieving data from {} to {} for {}'.format(pd.to_datetime(start, unit='ms'),
                                                            pd.to_datetime(end, unit='ms'), symbol))
        time.sleep(1.5)
    return data


# So let's give it a run and download the BTCUSD price data for the first 20 days in October 2019.  

# In[ ]:


# Define query parameters
pair = 'BTCUSD' # What is the currency pair we are interested in
bin_size = '1m' # This is the resolution at which we request the data
limit = 1000 # How many data points per call are we asking for
time_step = 1000 * 60 * limit # From the above calulate the size of each sub querry

# Fill in the start and end time of interest and convert it to timestamps
t_start = datetime.datetime(2019, 10, 1, 0, 0)
t_start = time.mktime(t_start.timetuple()) * 1000

t_stop = datetime.datetime(2019, 10, 20, 23, 59)
t_stop = time.mktime(t_stop.timetuple()) * 1000

# Create an bitfinex_api instance
api_v1 = bitfinex.bitfinex_v1.api_v1()

# Collect the data
pair_data = fetch_data(start=t_start, stop=t_stop, symbol=pair, interval=bin_size, tick_limit=limit, step=time_step)


# The data we receive may need to be cleaned, e.g. removal of error messages in case of connection problems.
# Also, the data is in a format that is not very convenient to use so we will in addition move everything to a pandas data frame.

# In[ ]:


# Remove error messages
ind = [np.ndim(x) != 0 for x in pair_data]
pair_data = [i for (i, v) in zip(pair_data, ind) if v]

# Create pandas data frame and clean data
names = ['time', 'open', 'close', 'high', 'low', 'volume']
df = pd.DataFrame(pair_data, columns=names)
df.drop_duplicates(inplace=True)
df.set_index('time', inplace=True)
df.sort_index(inplace=True)


# So now that we collected & cleaned the data we can have a look at the closing price by plotting it over time.

# In[ ]:


df.index = pd.to_datetime(df.index, unit='ms')

fig, ax = plt.subplots(1, 1, figsize=(18, 5))

ax.plot(df['close'])
ax.set_xlabel('date', fontsize=16)
ax.set_ylabel('BTC price [USD]', fontsize=16)
ax.set_title('Bitcoin closing price from {} to {}'.format(df.index[0], df.index[-1]))
ax.grid()

plt.show()


# Alright looks like we got what we wanted. So if we want to get data for a different time interval or currency pair we can easily do this now by changing the query parameters above accordingly.
# Now finally we can combine 

# In[ ]:


# Path to the old data from the 400+ crypto currency pairs at 1-minute resolution dataset
path_name = ('../input/392-crypto-currency-pairs-at-minute-resolution/cryptominuteresolution/btcusd.csv')

# Load the data
df_old = pd.read_csv(path_name, index_col='time')

# Convert timestamp to datetime
df_old.index = pd.to_datetime(df_old.index, unit='ms')

# Append the new data to the old data set
df_old = df_old.append(df)

# Remove duplicates and sort the data
df_old.drop_duplicates(inplace=True)
df_old.sort_index(inplace=True)

