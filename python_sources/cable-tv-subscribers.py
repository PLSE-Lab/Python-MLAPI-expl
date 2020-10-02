#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print("The file name that has the data is " ,os.listdir("../input"))


# Read the data into data frame

# In[ ]:


df = pd.read_csv("../input/CableTVSubscribersData.csv")
df.head()


# This data is quite simple , some simple statistics only can be drawn from this set

# 1. Segments  and the counts

# In[ ]:


segment = df['Segment'].value_counts(). reset_index()
segment.columns = ['Segment', 'Count'] # Changed the column names
plt.figure(figsize= (20,5)) # Make a plot size
trace = sns.barplot(x = segment['Segment'], y = segment['Count'], data = segment)
# Adding values on the top of the bars
for index, row in segment.iterrows():
    trace.text(x = row.name, y = row.Count+ 2, s = str(row.Count),color='black', ha="center" )
plt.show()


# In[ ]:


set = df[['Segment', 'subscribe']]
grouped = set.groupby(['Segment', 'subscribe']).size()
plt.figure(figsize= (20,5)) # Make a plot size
#trace = sns.barplot(x = segment['Segment'], y = segment['Count'], data = segment)
trace = grouped.plot(kind = 'bar')
grouped.unstack()


# Observation :  'SubNo'  count is more in each segment. It means number of  overall  subscriptions  are less.

# Any relation between income and the own house??. Explore this.

# In[ ]:


df[['income','ownHome' ]].groupby(['ownHome']).mean()


# Observation :  Average income for the ownHome category is more 

# In[ ]:




