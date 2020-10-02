#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../input/2018_03.csv")
for i in range(4,10):
    df = df.append(pd.read_csv("../input/2018_0"+str(i)+".csv"))
df.head()


# First we need to isolate only NJ Transit trips. (Amtrak schedules are not contained in the dataset).

# In[ ]:


df = df[df['type'] == 'NJ Transit']
df.head()


# Next, we will check for missing values.

# In[ ]:


df.isnull().sum()


# In[ ]:


df[df.isnull().any(axis=1)].head()


# We'll drop these missing values.

# In[ ]:


df = df.dropna()
df.describe()


# In[ ]:


df.describe(exclude=[np.number])


# Let's visualize the data

# In[ ]:


sns.distplot(df['delay_minutes'])


# As shown in the summary and the chart there are some outliers with very large delays.

# In[ ]:


sns.countplot(df['status'])


# In[ ]:


df[df['status'] == 'cancelled'].head()


# ## Where are these large delays and cancellations happening?
# 
# Let's look at lines, stops, origins, and destinations to see if there is any relationship with delays or cancellations

# In[ ]:


df['long_delay'] = df['delay_minutes'] > 5
df.groupby('line')['long_delay'].mean().sort_values(ascending=False).plot(kind='bar')


# In 7 of the 11 lines,  at least 20% of trips have long delays

# In[ ]:


x = df.groupby(['line', 'status']).size().unstack()
x['cancelled']/(x['departed']+x['estimated'])


# In[ ]:


x['cancelled']


# The cancel rate seems to be low for all trips. The Princeton Shuttle, unsurprisngly, has no cancellations.

# In[ ]:


ax = df.groupby('stop_sequence')["delay_minutes"].mean().plot()
ax.set_ylabel("average delay_minutes")


# There seems to be a slight relationship with delays increasing the further along the stop sequence.

# In[ ]:


df.groupby('from')['delay_minutes'].mean().sort_values(ascending=False).head(10)


# In[ ]:


df.groupby('to')['delay_minutes'].mean().sort_values(ascending=False).head(10)


# Lindenworld, Aberdeen-Matawan, Tuxedo, and Cherry Hill show up in both top 10 origins and destinations for longest delay so they may be worth investigating.

# ## When are these delays and cancellations happening?
# 
# Now, I'll compare date and time with delays and cancellations

# In[ ]:


df.date = pd.to_datetime(df.date)
x = df.groupby('date')['delay_minutes'].mean()
fig, ax = plt.subplots()
fig.set_size_inches(20,8)
fig.autofmt_xdate()
ax.plot(x)
ax.set_ylabel('average delay_minutes')
plt.show()


# In[ ]:


df.scheduled_time = pd.to_datetime(df.scheduled_time)
df['time'] = df.scheduled_time.dt.time

fig, ax = plt.subplots()
fig.set_size_inches(20,8)
x = df.groupby('time')['delay_minutes'].mean()
ax.plot(x)
ax.set_ylabel('average delay_minutes')
plt.show()


# In[ ]:


x.sort_values(ascending=False).head(5)


# Peak delays seem to be at around ~3:00am. 

# In[ ]:


x = df.groupby(['date', 'status']).size().unstack()
x = (x['cancelled']/(x['departed']+x['estimated']))

fig, ax = plt.subplots()
fig.set_size_inches(20,8)
ax.plot(x)
ax.set_ylabel('cancellation rate')
plt.show()


# 

# In[ ]:


x.sort_values(ascending=False).head(5)


# [A powerful storm hit New Jersey Transit](https://www.njtransit.com/sa/sa_servlet.srv?hdnPageAction=CustomerNoticeTo&NoticeId=2525) on March 8th which explains the extremely high cancellation rate on that day.
# ![](http://www.njtransit.com/%20/images-uploads/march%207%20storm.jpg)

# In[ ]:


x = df.groupby(['time', 'status']).size().unstack()
x = (x['cancelled']/(x['departed']+x['estimated']))

fig, ax = plt.subplots()
fig.set_size_inches(20,8)
ax.plot(x)
ax.set_ylabel('cancellation rate')
plt.show()


# In[ ]:




