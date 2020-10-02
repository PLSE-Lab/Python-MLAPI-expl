#!/usr/bin/env python
# coding: utf-8

# # Timeseries Analysis (v0.1)
# 
# This Kernel show basic analysis on basis of number of tweets/re-tweets per hour

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")


# ### Load CSV data
# 
# Encoding="ISO-8859-1" is required due to special characters in `text` column

# In[ ]:


df_tweets = pd.read_csv('../input/demonetization-tweets.csv', parse_dates=['created'], header=0, encoding="ISO-8859-1")
df_tweets.head(5)


# 
# ### Pre-process Data
# 
# We create new columns namely hour, date & count for our analysis

# **Plot graph for number of Re-Tweets per hour**

# In[ ]:


df_tweets['date'] = pd.DatetimeIndex(df_tweets['created']).date
df_tweets['hour'] = pd.DatetimeIndex(df_tweets['created']).hour
df_tweets['minutes'] = pd.DatetimeIndex(df_tweets['created']).minute
df_tweets['seconds'] = pd.DatetimeIndex(df_tweets['created']).second
#convertinng to seconds 
df_tweets['time'] = df_tweets['hour'] * 3600 + df_tweets['minutes'] * 60 + df_tweets['seconds']
df_tweets_filtered = df_tweets[['date', 'retweetCount','text','time']]
df_tweets_filtered.head(5)


# In[ ]:


#stripping text so hat it could be fit in the graph
label=df_tweets_filtered['text']
labels=label.head(10).str.split(' ').str
labels=labels[1]
labels


# In[ ]:


#scatter plot
N = 10
labels = labels
plt.subplots_adjust(bottom = 0.5)
plt.scatter(
    df_tweets_filtered['time'].head(10),df_tweets_filtered['retweetCount'].head(10), marker = 'o')
for label, x, y in zip(labels,df_tweets_filtered['time'].head(10), df_tweets_filtered['retweetCount'].head(10)):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (-30, 30),
        textcoords = 'offset points', ha = 'left', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'green', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.plot(df_tweets_filtered['time'].head(10), df_tweets_filtered['retweetCount'].head(10))
plt.show()


# In[ ]:





# In[ ]:




