#!/usr/bin/env python
# coding: utf-8

# ## Twitter Analysis as a Function of Time
# 
# Since every tweet also has a timestamp, we can do some temporal analysis of these tweets and visualiuze how the dynamics of the tweets unfold in time.
# 
# First, we will need to grab the day, month, and year that the tweets occur.  Then, we can visualize the data using the Pandas library.
# 
# 

# In[ ]:





# In[ ]:


#Import Libraries etc etc
import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy import signal
import re
get_ipython().run_line_magic('matplotlib', 'inline')

#Some nice plotting params
mpl.rcParams['figure.figsize'] = (8,5)
mpl.rcParams['lines.linewidth'] = 3
plt.style.use('ggplot')

#Read in the data.  Seems like the dates are the second last column
df = pd.read_csv('../input/tweets.csv', parse_dates = [-2])

def f(x): # I don't really care about the times that the tweets were made
    
    return dt.datetime(x.year,x.month,x.day)

df['time'] = df.time.apply( f)

time_series = pd.DataFrame(df.time.value_counts().sort_index().resample('D').mean().fillna(0))
time_series.columns = ['Tweets']

time_series.plot()


# Wow, what are those oscillations going on near January?  Lets take a closer look!

# In[ ]:







# In[ ]:





# In[ ]:


time_series = time_series.ix['2016-01-28':]
time_series.plot()
ts = time_series.values.ravel()


# That is really strange.  There are instances where there are more than 400 tweets on some days, and almost 0 on others.
# 
# I'm interested in the frequency of these oscillations.  That is to say: what is the time between two hughe spikes in tweets?
# 
# We can use tools from singal processing to examine the frequency.

# In[ ]:


fs = np.fft.fftfreq(np.arange(len(ts)).shape[-1])
x,y = signal.periodogram(ts)
plt.plot(x,y,'o-')

oneov = [1.0/j for j in range(2,8)]
plt.xticks(oneov,range(2,8))
ax = plt.gca()


ax.grid(True)
ax.set_yticklabels([])
ax.set_xlabel('No. of Days')


# This is a Periodogram, which tells us which frequency is the largest contrinutor.  The x axis is typically in 1/Days, so here, I have labeled where 2 through 7 days would lie on the x axis.
# 
# Wow, seems as if there is a big spike at 7.  This suggests every 7 days, a mass amount of tweets is made.  Lets try and figure on what weekday these tweets are made.
# 
# We will first need to find the places where the peaks occur.  Luckily, scipy has a function for that.

# In[ ]:


p = signal.find_peaks_cwt(ts, np.arange(1,4) )


t = np.arange(len(ts))
plt.plot(t,ts)
plt.plot(t[p],ts[p],'o')


# Looks like we got em!  Now, lets figure out on which weekday those occur.

# In[ ]:


r = time_series.ix[p].reset_index().copy()

r.columns = ['date','tweet']

r['weekday'] = r.date.apply(lambda x: x.weekday())

r.weekday.value_counts()


# The .weekday() function returns a number for the correpsonding weekday.  0 is monday, 6 is sunday, which means that 4 is Friday.
# 
# This shouldn't be a huge surprise.  Every friday, Muslims hold a congressional prayer known as Jumu'ah.

# In[ ]:




