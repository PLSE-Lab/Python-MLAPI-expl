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


# In[ ]:


data = pd.read_csv("../input/metro-bike-share-trip-data.csv")


# In[ ]:


data.info()


# In[ ]:


data.head()


# Lets put _ instead spaces in column names for easy coding.

# In[ ]:


data.columns = data.columns.str.replace(' ','_')
data.head()


# Change Start and End Time vlues to datetime format for easy analysis

# In[ ]:


data['Start_Time']=pd.to_datetime(data['Start_Time'])
data['End_Time']=pd.to_datetime(data['End_Time'])
data.info()


# Erase NaN values for clean data

# In[ ]:


data = data.dropna()
data.info()


# Change format of station and bike ID's to integer.

# In[ ]:


data['Starting_Station_ID'] = data.Starting_Station_ID.astype(int)
data['Ending_Station_ID'] = data.Ending_Station_ID.astype(int)
data['Bike_ID'] = data.Bike_ID.astype(int)
data.head()


# Number of stations

# In[ ]:


print(data.Starting_Station_ID.nunique())
print(data.Starting_Station_Latitude.nunique())
print(data.Starting_Station_Longitude.nunique())
print(data.Ending_Station_ID.nunique())
print(data.Ending_Station_Latitude.nunique())
print(data.Ending_Station_Longitude.nunique())


# number of bikes

# In[ ]:


data.Bike_ID.nunique()


# One way or Round Trip?

# In[ ]:


data.Trip_Route_Category.value_counts().plot(kind='bar')
plt.show()


# Pass types

# In[ ]:


data.Passholder_Type.value_counts().plot(kind='pie',autopct='%.2f')
plt.show()


# In[ ]:


data.Plan_Duration.value_counts().plot(kind  ='pie', autopct='%.2f')
plt.show()


# Total date collected dates

# In[ ]:


data.Start_Time.dt.date.nunique()


# Time of first and the last data

# In[ ]:


print(data.Start_Time.min())
print(data.Start_Time.max())


# The most used 5 bikes

# In[ ]:


data.Bike_ID.value_counts().head()


# In[ ]:


data.Starting_Station_ID.value_counts().head()


# In[ ]:


data.Ending_Station_ID.value_counts().head()


# In[ ]:


data.Duration.value_counts().head()


# In[ ]:


print(data.Duration.min())
print(data.Duration.max())


# In[ ]:


time_filter = data.Duration<2*60*60
data[time_filter].Duration.value_counts().plot(kind='bar', figsize=(10,10))
plt.show()


# When is the most common pick-up time?

# In[ ]:


data.Start_Time.dt.hour.value_counts().plot(kind='bar')
plt.show()


# * Most of the trips start after work hours (between 17.00 and 19.00)
# * Lunch time is also active (between 12.00 and 14.00)
# * It seems everybody sleeps after midnight, between 00.00 and 07.00 is the least active time.

# * Means of durations for different pass holders 
# * Pie chart fo total duration for different pass holders

# In[ ]:


Monthly = data.Passholder_Type == "Monthly Pass"
Walk = data.Passholder_Type == "Walk-up"
Flex = data.Passholder_Type == "Flex Pass"
fig = plt.figure
plt.subplot(1,2,1)
plt.bar(("monthly","walk-up","free"),[data[Monthly].Duration.mean(),data[Walk].Duration.mean(),data[Flex].Duration.mean()])
plt.subplot(1,2,2)
plt.pie([data[Monthly].Duration.sum(), data[Walk].Duration.sum(), data[Flex].Duration.sum()], labels = ("monthly","walk-up","free"), autopct='%.2f')
plt.show()


# Walk-up pass holders are leader at both means of a trip and total duration of trips.
# Monthly pass holders average trip time is less than others but total trip duration is more than free pass holders.

# Lets group duration levels
# * Less than 10 min => Very short
# * 10 min to 1 hour => Short
# * 1 hour to 4 hours => Mid
# * More than 1 hours =>Long

# In[ ]:


data["duration_level"] = ["long" if i>4*60*60 else "very short" if i<=10*60 else "short" if i<=60*60  else "mid" for i in data.Duration]
data.duration_level.value_counts().plot(kind = 'pie',autopct='%.2f')
plt.show()


# 1. Half of the trips take less than 10 minutes.
# 2. Long trips are less than 1% of all trips
# 
