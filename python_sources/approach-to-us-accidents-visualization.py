#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import calendar
sns.set()


# In[ ]:


month_names = dict(enumerate(calendar.month_abbr))
day_names = dict(enumerate(calendar.day_name))


# ### Load the dataset
# Load the data set and view the head(first 5 rows) of the dataset.

# In[ ]:


df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
df.head()


# View the columns in the dataframe

# In[ ]:


df.columns


# In[ ]:


df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])
df['Year'] = df['Start_Time'].dt.year
df['Hour'] = df['Start_Time'].dt.hour
df['Day'] = df['Start_Time'].dt.dayofweek
df['DayName'] = df['Start_Time'].dt.weekday_name
df['Month'] = df['Start_Time'].dt.month


# In[ ]:


df = df[(df['Year']>2015) & (df['Year']<2020)]


# Looking at the variables, we can use columns like the Start_Time, Wind_Speed, City, Distance, Weather Condition etc.

# ### Accident Frequency per Time
# We can show at what time does accidents commonly happen. First we'll start by showing the frequency by differnt time granularity (hours, days months, years

# In[ ]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
fig.tight_layout(pad=3)

hours = df.groupby('Hour').size()
ax1.bar(hours.index, hours)
ax1.set_xticks(hours.index)
ax1.set_xlabel('Factor (Hour)')

days = df.groupby('Day').size()
ax2.bar(days.index, days)
ax2.set_xticks(days.index)
ax2.set_xticklabels([day_names[i] for i in days.index])
ax2.set_xlabel('Factor (Day)')

months = df.groupby('Month').size()
ax3.bar(months.index, months)
ax3.set_xticks(months.index)
ax3.set_xticklabels([month_names[i] for i in months.index])
ax3.set_xlabel('Factor (Month)')

years = df.groupby('Year').size()
ax4.bar(years.index, years)
ax4.set_xticks(years.index)
ax4.set_xlabel('Factor (Year)')


# * The graph with the hours factor shows that accidents rise from 4 AM, peaking from 7 to 8 AM(close to the traditional hours 8 to 9 AM). As seen from the graph, accidents rise again from 16 to 17(4 to 5 PM, a time where people check out from work and go home).
# * From the Day graph, accidents occur the most during the weekdays.
# * From the Month graph, we can see that the accidents rise on the month of August onwards.
# * Finally from the Year graph, we can notice that there is a rise of accidents from year 2016 to 2019

# ### Accident Severity based on location

# In[ ]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
fig.tight_layout(pad=3)

severity_1 = df[df['Severity']==1]
sns.scatterplot(x ='Start_Lng',y='Start_Lat', data=severity_1, hue='Severity',linewidth=0, ax=ax1)

severity_2 = df[df['Severity']==2]
sns.scatterplot(x ='Start_Lng',y='Start_Lat', data=severity_2, linewidth=0,ax=ax2)

severity_3 = df[df['Severity']==3]
sns.scatterplot(x ='Start_Lng',y='Start_Lat', data=severity_3,linewidth=0,ax=ax3)

severity_4 = df[df['Severity']==4]
sns.scatterplot(x ='Start_Lng',y='Start_Lat', data=severity_4,linewidth=0,ax=ax4)


# In[ ]:


plt.figure(figsize=(18, 8))
sns.scatterplot(x ='Start_Lng',y='Start_Lat', data=df, linewidth=0, hue='State')


# Using scatterplot we were able to plot the accidents and observe areas where accidents mostly happens. Regardless of state, accidents happen in every corner of US. Even in severity, levels doesn't differ that much from each other except level 1.

# In[ ]:


plt.figure(figsize=(14, 8))
sns.countplot(x='State', data=df, order=df['State'].value_counts().iloc[:10].index)


# Showing the top 10 states with the highest count of accidents, CA state topped the list.

# ### Weather Condition and Accidents
# Weather condition is also one of the factors that contributes to accidents. We'll see if the accidents were related to the weather.

# In[ ]:


plt.figure(figsize=(14, 8))
sns.countplot(y='Weather_Condition', data=df, order=df['Weather_Condition'].value_counts().iloc[:10].index)


# Observing the top 10 weather conditions. Accidents happen mostly on a clear weather condition followed by cloud related weather coditions. This may also tell that weather might not be the best contributing factor in the accidents that happens in the US.

# Any feedback would be appreciated for this notebook. Thanks :)

# In[ ]:




