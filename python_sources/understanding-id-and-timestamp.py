#!/usr/bin/env python
# coding: utf-8

# # Relationships between id and timestamp columns

# In[ ]:


# dependencies
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
p = sns.color_palette()


# In[ ]:


# load data
hdf = pd.HDFStore("../input/train.h5")
df = pd.concat([hdf.select(key) for key in hdf.keys()])
hdf.close()


# Ids might be stocks since 1424 unique ids is in the range of a stock universe.
# 

# In[ ]:


print('Unique Ids: ', df['id'].nunique()) # this is about the size of a typical stock selection universe


# The number of ids has peculiar behavior over time.  They roll off and then jump again with an upward trend.  Also, at no point are all ids present.

# In[ ]:


# id counts w.r.t time
temp = df.groupby('timestamp').apply(lambda x: x['id'].nunique())
plt.figure(figsize=(8,4))
plt.plot(temp, color=p[0])
plt.xlabel('timestamp')
plt.ylabel('id count')
plt.title('Number of ids over time')


# About 500 ids are present the entire time series.  Some ids have hardly any data (minimum of 2 timestamps!).  

# In[ ]:


# lifespan of each id
temp = df.groupby('id').apply(len)
temp = temp.sort_values()
temp = temp.reset_index()
plt.figure(figsize=(8,4))
plt.plot(temp[0], color=p[0])
plt.xlabel('index for each id sorted by number of timestamps')
plt.ylabel('number of timestamps')
plt.title('Number of timestamps ("Lifespan") for each id')
print(temp[0].describe())


# Many of the shortest lived ids fall at the very beginning or end of time.  Each id appears to have a continuous subset of timestamps.

# In[ ]:


N= 100
temp2 = df[df['id'].isin(temp['id'].head(N).values)]
temp2 = temp2.sort_values(['id', 'timestamp'])
temp2 = temp2.pivot(index='timestamp', columns='id', values='id')
plt.figure(figsize=(8,4))
plt.plot(temp2)
plt.xlabel('timestamp')
plt.ylabel('id')
plt.title('"Lifespan" for the {} shortest lived ids'.format(N))


# It seems many ids have the first or last timestamp when we look at the ids with the median number of timestamps. 

# In[ ]:


n_start = 700
n_end = 750
temp2 = df[df['id'].isin(temp['id'][n_start:n_end].values)]
temp2 = temp2.sort_values(['id', 'timestamp'])
temp2 = temp2.pivot(index='timestamp', columns='id', values='id')
plt.figure(figsize=(8,4))
plt.plot(temp2)
plt.xlabel('timestamp')
plt.ylabel('id')
plt.title('"Lifespan" for ids ranked from {}-{}'.format(n_start, n_end))


# Confirmation that the starting and ending timestamp is not random.  

# In[ ]:


print('Ids with timestamp=0: ', len(df[df['timestamp'] == 0]))
print('Ids with timestamp=max: ', len(df[df['timestamp'] == df['timestamp'].max()]))
print('Total ids: ', df['id'].nunique())

