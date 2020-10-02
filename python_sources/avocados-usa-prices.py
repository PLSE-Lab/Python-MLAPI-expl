#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Avocado prices in the USA

# In[4]:


df = pd.read_csv('../input/avocado.csv')


# In[5]:


df.head()


# In[6]:


df['region'].unique()


# In[7]:


df['year'].unique()


# In[8]:


df['type'].unique()


# In[9]:


df.tail()


# ## Houston:

# When I moved from Texas to Indiana, I noticed the prices for avocados were higher in Indiana. Let's see if this data set supports my experience.

# In[11]:


hou = df.loc[df['region']=='Houston',]


# In[12]:


hou.head()


# In[13]:


g = sns.catplot(data=hou, kind='box', x='year', y='AveragePrice', palette='winter')
g.fig.suptitle("Houston: Avocado prices")


# In[14]:


g = sns.catplot(data=hou, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')
g.fig.suptitle("Houston: Average avocado prices")


# In[15]:


g = sns.catplot(data=hou, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')
g.fig.suptitle("Houston: Average Avocado prices")


# ## Indianapolis:

# In[16]:


indy = df.loc[df['region']=='Indianapolis',]


# In[17]:


indy.head()


# In[18]:


g = sns.catplot(data=indy, kind='box', x='year', y='AveragePrice', palette='winter')
g.fig.suptitle("Indianapolis: Avocado prices")


# In[19]:


g = sns.catplot(data=indy, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')
g.fig.suptitle("Indianapolis: Average avocado price")


# In[20]:


g = sns.catplot(data=indy, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')
g.fig.suptitle("Indianapolis: Average avocado price")


# Since I actually live in northwest Indiana I wanted to check a couple of nearby cities and see how the prices compare. 

# ## Chicago:

# In[21]:


chicago = df.loc[df['region']=="Chicago",]
chicago.head()


# In[22]:


g = sns.catplot(data=chicago, kind='box', x='year', y='AveragePrice', palette='winter')
g.fig.suptitle("Chicago: Avocado prices")


# In[23]:


g = sns.catplot(data=chicago, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')
g.fig.suptitle("Chicago: Average avocado price")


# In[24]:


g = sns.catplot(data=chicago, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')
g.fig.suptitle("Chicago: Average avocado price")


# ## Grand Rapids:

# In[25]:


GrandRapids = df.loc[df['region']=="GrandRapids",]
GrandRapids.head()


# In[26]:


g = sns.catplot(data=GrandRapids, kind='box', x='year', y='AveragePrice', palette='winter')
g.fig.suptitle("Grand Rapids: Avocado prices")


# In[27]:


g = sns.catplot(data=GrandRapids, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')
g.fig.suptitle("Grand Rapids: Average avocado price")


# In[28]:


g = sns.catplot(data=GrandRapids, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')
g.fig.suptitle("Grand Rapids: average avocado price")


# - This data supports my experience that I was paying more for avocados in Indiana than I was in Texas. 
# - A quick Google search reveals 2017 experienced an avocado shortage, although the price increase appears to have hit some areas harder than others. 

# # 2017 data:

# In[29]:


yr2017 = df.loc[df['year'].isin(['2017'])]
yr2017.head()


# In[30]:


g = sns.catplot(data=yr2017, kind='swarm', palette='magma', x='type', y='AveragePrice', hue='region')


# ## Total US:

# In[31]:


totalUS = df.loc[df['region'].isin(['TotalUS'])]
totalUS.head()


# In[32]:


g = sns.catplot(data=totalUS, kind='box', x='year', y='AveragePrice', palette='winter')
g.fig.suptitle("Total US: Avocado prices")


# In[33]:


g = sns.catplot(data=totalUS, kind='swarm', x='year', y='AveragePrice', hue='type', palette='winter')
g.fig.suptitle("Total US: Average avocado price")


# In[34]:


g = sns.catplot(data=totalUS, kind='swarm', palette='hls', x='type', y='AveragePrice', hue='year')
g.fig.suptitle("Total US: average avocado price")

