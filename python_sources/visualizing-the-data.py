#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set()

get_ipython().system('conda install pandas=0.22')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Reading the data. 
# I chose PM 2.5 as it is the most dangerous of all

# In[57]:


df = pd.read_csv('../input/PM25.csv')
df.tail()


# ## Cleaning the data
# Setting the time column as index

# In[58]:


df['time'] = pd.to_datetime(df['time'], yearfirst=True)
df.set_index('time', inplace=True)
df.drop('Unnamed: 0', axis=1, inplace=True)
df.index.name = 'Time'
df.info()


# In[59]:


df.head()


# We have lots of null values, so we will remove all rows where we **only** have nulls 

# In[60]:


df.dropna(how='all', inplace=True)
df.head()


#  ### We will plot the hourly values groupped by weekday for every month

# In[61]:


data_weekday = df.groupby([ 
    df.index.month, 
    df.index.weekday_name,
    df.index.hour]).apply(np.mean)
data_weekday.index.rename(names=['month','weekday','hour'], inplace=True)
data_weekday


# In[ ]:


fig = data_weekday.unstack(level=0).unstack(level=0).dropna(axis=1, how='all')    .plot(subplots=True, figsize=(20,300), kind='line')[0].get_figure()
fig.savefig('weekday.pdf')


# Just with a quick look at this graph shows us that the highest pollution happens at night

# ### Plotting the data for every month*
# *average for every n-th month of all the years that we have data for

# In[ ]:


data_month = df.groupby([ 
    df.index.month, 
    df.index.day]).apply(np.mean)
data_month.index.rename(names=['month','day'], inplace=True)
data_month.head()


# In[ ]:


fig = data_month.unstack(level=0).dropna(axis=1, how='all')    .plot(subplots=True, figsize=(20,300), kind='line')[0].get_figure()
fig.savefig('monthly.pdf')


# ### Plo the data for every year

# In[ ]:


data_year = df.groupby([ 
    df.index.year, 
    df.index.month,
    df.index.day]).apply(np.mean)
data_year.index.rename(names=['year','month','day'], inplace=True)
data_year.head()


# In[ ]:


fig = data_year.unstack(level=0).dropna(axis=1, how='all')    .plot(subplots=True, figsize=(20,300), kind='line')[0].get_figure()
fig.savefig('yearly.pdf')


# In[ ]:


fig = data_year    .plot(subplots=True, figsize=(20,50), kind='line')[0].get_figure()
fig.savefig('all.pdf')


# In[ ]:




