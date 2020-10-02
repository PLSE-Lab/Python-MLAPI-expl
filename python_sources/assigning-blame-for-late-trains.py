#!/usr/bin/env python
# coding: utf-8

# # SEPTA On Time Performance - If a train is late in the city does anyone notice?
# 
# In which I look at something that has been bothering me about the SEPTA dataset.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.plt.rcParams['figure.figsize'] = (12, 10)


# In[ ]:


from sqlalchemy import create_engine
con = create_engine('sqlite:///../input/database.sqlite')


# In[ ]:


df = pd.read_sql_table('otp', con)
df.head()


# In[ ]:


df.info()


# In[ ]:


df.loc[df.status=="1440 min", "status"] = "999 min"
df['status_n'] = df.status.str.replace("On Time", "0").str.replace(" min","").astype("int")


# Now to look at just line 550, to cut down on the data.

# In[ ]:


t = df[df.train_id=="550"].sort_values(by='timeStamp')
t.head()


# How does the lateness of this train line evolve over time?

# In[ ]:


df[df.train_id=="550"].sort_values(by='timeStamp').iloc[:100].plot(x='date', y='status_n')


# Does that seem right to you? That initial 10 minute lateness makes the subsequent stations seem very poor indeed, even as some of them actually made up some of the late time. What about the lateness contribution of each individual station?

# In[ ]:


t['status_diff']= t.status_n.diff()
t.head()


# In[ ]:


t.plot(x='date', y='status_diff')


# 

# In[ ]:


t.status_diff.hist(bins=50, log=True)


# 

# In[ ]:


tg = t.groupby(['next_station']).mean().sort_values(['status_diff'])
tg


# 

# In[ ]:


tg.plot(kind="scatter", x='status_n', y='status_diff')


# In[ ]:


tg.corr()


# 

# In[ ]:


df.sort_values(by=['train_id', 'timeStamp'], inplace=True)


# In[ ]:


df['status_diff'] = df.status_n.diff()


# In[ ]:


df.loc[df.next_station == "None",'status_diff'] = np.NaN
df.head()


# In[ ]:


diffs = df.dropna().groupby(['next_station']).mean().sort_values(['status_diff'])
diffs


# Looking at data in terms of _incremental_ lateness makes a big difference in what stations seem to be problematic. I think this should be a better indicator of problem areas in the transportation system. Incremental lateness in any station seems to have a cumulative effect on the lateness of subsequent stations.

# In[ ]:


diffs.plot(kind='scatter', x='status_n', y='status_diff')


# In[ ]:


diffs.corr()


# This way of looking at the data is even more valuable when looking at more than one train line, it seems.
# 
# I think that in the future I can do some useful things with this.
