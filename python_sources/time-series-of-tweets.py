#!/usr/bin/env python
# coding: utf-8

# ## A Time Series of Pro-Isis Tweets
# 
# It is really easy to see how many tweets were made on a particular calender day.
# 
# 

# In[ ]:


import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 4

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')

df = pd.read_csv('../input/tweets.csv', parse_dates = [-2])

def f(x): # I don't really care about the times that the tweets were made
    
    return dt.datetime(x.year,x.month,x.day)

df['time'] = df.time.apply( f)

time_series = pd.DataFrame(df.time.value_counts().sort_index().resample('D').mean().fillna(0))
time_series.columns = ['Tweets']
#How many tweets on each day.  Resample to get missing days

fig, ax = plt.subplots(figsize = (16,8))

time_series.plot(ax = ax)

time_series.rolling(window=7).mean().plot(ax = ax, linewidth = 2,color = 'k')


brussels = '2016-03-22'
paris = '2015-11-13'


ax.plot(brussels,183,color = 'w',
        marker = 'v',
        markersize = 12, 
        linestyle = '',
       markeredgewidth = 2
       )

ax.plot(paris,0,color = 'w',
        marker = 'o',
        markersize = 12, 
        linestyle = '',
        markeredgewidth = 2
        )




ax.margins(None,0.1)
ax.legend(['Tweets Made','Rolling 7 Day Mean', 'Bombing in Brussels','Shooting in Paris'], loc = 'upper left', numpoints = 1)
ax.set_xlabel('Date')
ax.set_ylabel('No of Tweets')


# Looks like the tweets really start to pick up around Janurary 2016.

# In[ ]:




