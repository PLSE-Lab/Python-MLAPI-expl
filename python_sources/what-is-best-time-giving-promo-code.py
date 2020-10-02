#!/usr/bin/env python
# coding: utf-8

# **What is the best time to give a promo code in Airbnb?**
# 
# Sometimes, A company give a promo code to increase customer's demand and caused increase income and profit. 
# Customer's demand ussually being down in particular pattern period, which is called low season.
# When is the low season occurred?
# And when is the high season occurred?
# To analyze it, we must prepare the data. 
# We use calendar table that is contain of list available room on every day to determine low or high season. 

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


dfc = pd.read_csv('../input/calendar.csv')
dfl = pd.read_csv('../input/listings.csv')
dfc.head(10)


# Before we processing and interpreted data, we must do prepare and profiling data. 
# 1. Look and analyze data. We have to know what will we do and design to give a new insight business. In that case, I count available / non available room to determine low/high season demand. 
# 2. Check start and end period to determine the sample of research
# 3. Check unique key of table. Check whether is there any listing room who has two condition (available and non available) in the same day
# 4. Check each of variable, control variable, response variable, and influence factor. in that case, we must check whether is there any change of total quantity listing room in the sample period. 

# In[4]:


# Check start and end period to determine the sample of research
dfc.min()


# In[5]:


dfc.max()


# Period data start from 2016-01-04 till 2017-01-02. Because of range period data in one year, so we do  monthly analysis. We use 12 sample data month, that is Jan - Dec 2016. 

# In[6]:


#Check whether is there any listing room who has two condition (available and non available) in the same day
dfcek = dfc.groupby(['date','listing_id']).count()
dfcek[dfcek['available']>1]


# There is no data listing room that has two condition / two rows in the same day. So that the unique key is listing_id and date

# In[7]:


# check whether is there any change of total quantity listing room in the sample period
dfc.groupby('date').count().head(5)


# In[8]:


dfc.groupby('date').count().tail(5)


# In[9]:


dfc.listing_id.nunique()


# The listing room has the same quantity on every day. It is proved by count listing_id in the sample date. If we count distinct listing_id for all data, we have the same quantity too. So we can called the total of quantity listing room as control variable

# The next step is preparing data for analyze. We take the data which is has available = false. It means the room is booked in that day. We count booking room in a day, and do analyze monthly and weekly. Monthly analysis is analysis for determining what is the date when low season happening. Weekly analysis is analysis for determining what is the day in weekly when low season happening. 

# In[11]:


#prepare data
dfc['date'] = pd.to_datetime(dfc['date'])
dfc1 = dfc[dfc['available']== 'f']
dfc2 = dfc1.groupby('date')['listing_id'].count().reset_index()
dfc2['date'] = pd.to_datetime(dfc2['date'])
dfc2['Month'] = dfc2['date'].dt.strftime('%b')
dfc2['Day'] = dfc2['date'].dt.day   
dfc2.set_index(['Month','Day']).sum(level=[0,1]).reset_index()
dfc2[dfc2['Month']=='Jan']


# Afer taht, make a visualization to analyze data. Make a sure that data is ready. 

# In[13]:


#design a visualization. Make a chart (a whole data to know about the trend of data)
plt.figure(figsize=(15, 8))
plt.plot(dfc2.date, dfc2.listing_id, color='b', marker='*', linewidth=0.3)
plt.title("Count of Booking Room Demand")
plt.xlabel('Date')
plt.ylabel('Count listing_id')


# After look the chart, we try divide the analyze into three parts. So lets call the part is cluster I, II, and III.

# In[14]:


#Prepare data for cluster based on period.
dfc2['date'] = pd.to_datetime(dfc2['date'])
dfc2["weekday"] = dfc2["date"].dt.weekday_name
dfc3 = dfc2[(dfc2['date']>='2016-01-01')&(dfc2['date']<='2016-04-01')]
dfc4 = dfc2[(dfc2['date']>'2016-04-01')&(dfc2['date']<'2016-07-01')]
dfc5 = dfc2[(dfc2['date']>='2016-07-01')&(dfc2['date']<='2017-01-02')]
dfc2.head(5)


# In[15]:


#all data
plt.figure(figsize=(15, 8))
plt.scatter(dfc2.weekday, dfc2.listing_id, color='b', marker='.')
plt.title("Average listing price by date")
plt.xlabel('date')
plt.ylabel('average listing price')
plt.grid()


# In[17]:


plt.figure(figsize=(10, 6))
sns.boxplot(x = 'weekday',  y = 'listing_id', data = dfc2, palette="Blues", width=0.4)
plt.title("Count of Booking Room Demand Weekly Analysis")
plt.xlabel('Name of Day')
plt.ylabel('Count of Listing_id')
plt.show()


# In[19]:


plt.figure(figsize=(10, 6))
sns.boxplot(x = 'weekday',  y = 'listing_id', data = dfc3, palette="Blues", width=0.4)
plt.title("Count of Booking Room Demand Weekly Analysis Cluster I")
plt.xlabel('Name of Day')
plt.ylabel('Count of Listing_id')
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
sns.boxplot(x = 'weekday',  y = 'listing_id', data = dfc4, palette="Blues", width=0.4)
plt.title("Count of Booking Room Demand Weekly Analysis Cluster II")
plt.xlabel('Name of Day')
plt.ylabel('Count of Listing_id')
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
sns.boxplot(x = 'weekday',  y = 'listing_id', data = dfc5, palette="Blues", width=0.4)
plt.title("Count of Booking Room Demand Weekly Analysis Cluster III")
plt.xlabel('Name of Day')
plt.ylabel('Count of Listing_id')
plt.show()


# When we look each of chart, there is not enough to determine the low season in each cluster. If we combine them, we take all data (the first boxplot), it is easier to find low demand. The low demand is found on Wednesday. 

# In[22]:


#do montlhy analysis. Create visualization in monthly.
plt.figure(figsize=(15, 8))
sns.boxplot(x = 'Day',  y = 'listing_id', data = dfc2, palette="Purples", width=0.4)
plt.title("Count of Booking Room Demand Monthly Analysis")
plt.xlabel('Name of Day')
plt.ylabel('Count of Listing_id')
plt.show()


# There are a lot anomamly data or outayer data. We can filter it, to analyze deeper and look the chart more detail. 

# In[23]:



dfc7 = dfc2[dfc2['listing_id']<1500]
plt.figure(figsize=(15, 8))
sns.boxplot(x = 'Day',  y = 'listing_id', data = dfc7, palette="Purples", width=0.4)
plt.title("Count of Booking Room Demand Monthly Analysis")
plt.xlabel('Name of Day')
plt.ylabel('Count of Listing_id')
plt.show()


# In[24]:


#cluster I
plt.figure(figsize=(15, 8))
sns.boxplot(x = 'Day',  y = 'listing_id', data = dfc3, palette="Purples", width=0.4)
plt.title("Count of Booking Room Demand Monthly Analysis Cluster I")
plt.xlabel('Name of Day')
plt.ylabel('Count of Listing_id')
plt.show()


# In[25]:


#Cluster II
plt.figure(figsize=(15, 8))
sns.boxplot(x = 'Day',  y = 'listing_id', data = dfc4, palette="Purples", width=0.4)
plt.title("Count of Booking Room Demand Monthly Analysis Cluster II")
plt.xlabel('Name of Day')
plt.ylabel('Count of Listing_id')
plt.show()


# In[26]:


dfc7 = dfc2[dfc2['listing_id']<1500]
plt.figure(figsize=(15, 8))
sns.boxplot(x = 'Day',  y = 'listing_id', data = dfc5, palette="Purples", width=0.4)
plt.title("Count of Booking Room Demand Monthly Analysis Cluster III")
plt.xlabel('Name of Day')
plt.ylabel('Count of Listing_id')
plt.show()


# When we look each of chart, there is not enough to determine the low season in each cluster. If we combine them, we take all data (the first boxplot), it is easier to find low demand. The low demand is found on the range date 16 - 24 of each month. 
