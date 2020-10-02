#!/usr/bin/env python
# coding: utf-8

#  **I have noticed in many situations, especially time series forecast, that it is advised to transform the data using logarithms, especially when dealing with trends; one of the purposes is to stabilize the variance. This notebook compares a series to its logged version and studies variance to observe if, in time, the variance becomes stable.**

# Firstly, let's take a look at the data; and the focus will be on the Period and Revenue variables.

# In[ ]:



import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/time-series-starter-dataset/Month_Value_1.csv')
data.head()


# In[ ]:


print('\n data types:')
data.dtypes


# The Revenue variable is expressed in millions, so I'm creating a new variable so it will be easier to follow:

# In[ ]:


data["Revenue - millions -"] = data["Revenue"]/1000000
data.head()


# Let's see how the revenue series looks like:

# In[ ]:


import matplotlib.dates as mdates

from datetime import datetime

fig = plt.figure()
ax = fig.add_subplot(111)

ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
x = data['Period']
ts = data['Revenue - millions -'] 
#ax.plot(data['Period'], data['Revenue/10000'])
ax.plot(x, ts)
fig.autofmt_xdate()

ax.grid(True)

plt.show()


# Now let's log it and graph the logged version:

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)

ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
x = data['Period']
ts_log = np.log(ts) 
#ax.plot(data['Period'], data['Revenue/10000'])
ax.plot(x, ts_log)
fig.autofmt_xdate()

ax.grid(True)

plt.show()


# Ok, so graphically it's noticeable that the series seems more balanced. Let's see if the variance is really stabilising or not. I'm transforming it into a dataframe and then calculating the variance at each new point in the series.

# In[ ]:


ts_log_df = pd.DataFrame(ts_log)
ts_log_df['Variance'] = 1


# In[ ]:


for i in ts_log_df.index:
    ts_log_df['Variance'].iloc[i] = ts_log_df['Revenue - millions -'].iloc[0:i].var()


# In[ ]:


ts_log_df.head(20)


# Now let's graph the variance:

# In[ ]:


nb = np.arange(0,96)
plt.plot(nb,ts_log_df['Variance'])


# So it's true, after a while the variance of the logged series stabilizes.
