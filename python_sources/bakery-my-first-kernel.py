#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **1. Load and examine data**

# In[ ]:


df = pd.read_csv('../input/BreadBasket_DMS.csv')
df.head()


# A quick look at the data table shows that the Item column has quite a few values listed as NONE - leaving this in for now as it still contributes to transactions under the assumption that it was an item of some kind.

# In[ ]:


df.info()


# Creating a datetime column and a column to show the day of the week.

# In[ ]:


df['datetime'] = pd.to_datetime(df['Date']+' '+ df['Time'])
df['day_of_week'] = df['datetime'].dt.day_name()


# **2. Data Exploration**
# 
# OK, let's start having a closer look at the data! We've sold a total of 21,293 items in 9,531 transactions. That's 2.2 items per transaction.

# In[ ]:


total_items = df['Transaction'].count()
unique_transactions = df['Transaction'].nunique()

items_per_transaction = total_items / unique_transactions
print(total_items, unique_transactions,items_per_transaction)


# Broken down by day of the week we see that Saturday is our best day both for total items sold and transactions, but we're selling slightly more items per transaction on a Sunday.

# In[ ]:


new_df = pd.pivot_table(df, values='Transaction', columns='day_of_week', aggfunc=('count','nunique')).transpose().reset_index()
new_df = new_df.sort_values(by='count', ascending=False)


# In[ ]:


total_items_by_weekday = df['day_of_week'].value_counts()
print(total_items_by_weekday.sort_values(ascending=False))


# In[ ]:


unique_transactions_by_weekday = df.groupby('day_of_week')['Transaction'].nunique()
print(unique_transactions_by_weekday.sort_values(ascending=False))


# In[ ]:


items_per_transaction_by_weekday = total_items_by_weekday / unique_transactions_by_weekday
print(items_per_transaction_by_weekday.sort_values(ascending=False))


# In[ ]:


new_df.plot.bar('day_of_week')


# But what are our bestseller? Unsurprisingly, it's coffee and bread, followed by tea and cake! But our little bakery sells a lot of things - 95 different items, to be precise.

# In[ ]:


df['Item'].value_counts().head(20)


# In[ ]:


df['Item'].nunique()


# In[ ]:


list_of_items = df['Item'].unique().tolist()
print(list_of_items)


# **3. Time series analysis**
# 
# Finally, let's have a look at the bakery's transactions over time. It looks like the bakery had a really good launch week and then transactions trailed off, hitting a low beginning of January (maybe due to the bakery being close over the Christmas/NY period?) and then business picked up again.

# In[ ]:


df.set_index('datetime', inplace=True, drop=True)


# In[ ]:


df_ts = df.drop(['Date', 'Time', 'day_of_week', 'Item'], axis=1)
weekly = df_ts.resample('W-MON').agg(['count', 'nunique'])
weekly['Transaction'].plot(kind='line')


# In[ ]:


monthly = df_ts.resample('M').agg(['count', 'nunique'])
monthly['Transaction'].plot(kind='line')

