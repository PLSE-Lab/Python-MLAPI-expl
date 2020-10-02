#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## **Read file**

# In[ ]:


df = pd.read_csv('../input/BreadBasket_DMS.csv')


# ## Exploratory analysis

# In[ ]:


df.head(10)


# **RRC observation**: Huh, interesting. A dataframe line is a transaction or a part of it. So, same transaction number implies is part of the same payment. See line 0 (Transaction=1) vs lines 1 and 2 (Transaction=2)

# In[ ]:


print("Length of the dataset: %d\nNumber of different items: %d" % (len(df), len(df.Item.unique())))


# In[ ]:


print("Number of different transactions: %d" % df['Transaction'].max())


# **RRC observation**: Ok, almost 10k different transaction over a total of 21k lines of dataset. Expecting ~2 items per transaction as average, right? Let's confirm. 

# In[ ]:


nTransactions = df.groupby(['Transaction'])['Item'].count().reset_index()
nTransactions.columns = ['Transaction', 'nItems']


# In[ ]:


#Checking...
nTransactions.head()


# In[ ]:


print("Number of Items per transaction in average: %.2f" % nTransactions['nItems'].mean())


# **RRC**: Done! Ok, but let's dig a bit more... 

# In[ ]:


nTransactions['nItems'].describe()


# **RRC**: Whoa! 12 is the max. And nice quartiles as well... Such a great information to work with.  

# 
# **RRC**: Ok, first ideas.
# * Let's see the distribution of the transactions along the time. First, hour of the day. Then we will calculate the day of the with corresponding to the date and see what do we have. 
# * Evaluate which items are most demanded. 
# * Evaluate those Items in time. 

# In[ ]:


#Transforming original dataset. First, adding the new column "number of Items, nItems" to the original. Just joining by Transaction .
# tt stands for "transformed Transactions" :o)
#On the other hand, we build 'times' dataset for time analysis

tt = nTransactions.merge(df, on='Transaction')
times = tt.drop_duplicates(subset='Transaction', keep='first')[['Transaction', 'nItems', 'Date', 'Time']]
times.head()


# In[ ]:


timeCount = times.groupby('Time').count()['Date'].reset_index()
timeCount.head()
#Ok, let's group by a more generic hour. What about hour and decimal of minutes? ;)


# In[ ]:


MIN_UNIT_INDEX=4
HOUR_UNIT_INDEX=2
times['hourMin'] = times['Time'].apply(lambda x: str(x)[:HOUR_UNIT_INDEX]+"x")
times.head()


# In[ ]:


timeCount = times.groupby('hourMin').count()['Date'].reset_index()
timeCount.columns=['hourMin', 'countTransactions']

timeCount.head(10)


# In[ ]:


len(timeCount)


# In[ ]:


timeCount.describe()


# In[ ]:


timeCount.plot.bar(x='hourMin', y='countTransactions', rot=30, figsize=(15,10))


# **RRC**: Now, same analysis by day of the week. 

# In[ ]:


times['dayWeek'] = pd.to_datetime(times['Date']).dt.weekday_name


# In[ ]:


times.head()


# In[ ]:


timesDay = times.groupby('dayWeek').count()['Transaction'].reset_index()
timesDay.columns = ['dayWeek', 'nTransactions']


# In[ ]:


timesDay.plot.bar(x='dayWeek', y='nTransactions', rot=30, figsize=(15,10))


# **RRC**: Now, let's join this dataframe with the original in order to get this info at line level

# In[ ]:


times = times[['Transaction', 'nItems', 'hourMin', 'dayWeek']]
dfplus = df.merge(times, on='Transaction')


# In[ ]:


dfplus.head()


# In[ ]:


res = dfplus.groupby(['dayWeek', 'Item']).count()['nItems'].reset_index()


# In[ ]:


res


# In[ ]:


mostPerDay = res.groupby('dayWeek').agg(['min', 'max'])


# In[ ]:


mostPerDay


# **RRC**: Cool! Let's calculate top of the items selled

# In[ ]:


totalItems = res.groupby('Item').sum().sort_values(by='nItems', ascending=False)


# In[ ]:


# Top 10 ;P
totalItems[:10]


# ## Some conclusions
# 
# * The top Item is Coffee (of course!)
# * But the max items selled per day are the Vegan mincepie and the Victorian Sponge
# * Would be nice to see how the sellings of this items evolve on time. 

# In[ ]:




