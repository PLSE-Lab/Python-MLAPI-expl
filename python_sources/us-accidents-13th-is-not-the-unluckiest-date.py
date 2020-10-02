#!/usr/bin/env python
# coding: utf-8

# ## Some basic exploratory data analysis on the US accident data

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
sns.set()


# In[ ]:


df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')


# In[ ]:


print(df.info())


# In[ ]:


counts_by_severity = df.groupby('Severity').count()['ID']


# ### Most accidents seem to be of severity 2 category

# In[ ]:


counts_by_severity.plot(kind='bar')


# In[ ]:


counts_by_severity.plot(kind='pie')


# In[ ]:


df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month_name()
df['Day'] = df['Start_Time'].dt.day
df['Day_name'] = df['Start_Time'].dt.day_name()
df['Hour'] = df['Start_Time'].dt.hour


# ## Accidents by hour (weekdays and weekends)

# In[ ]:


accidents_by_hour = df.groupby('Hour').count()['ID']
accidents_by_hour.plot(kind='bar', figsize=(10, 8))


# In[ ]:


weekdays_df = df[(df['Day_name'] == 'Monday') | (df['Day_name'] == 'Tuesday') | (df['Day_name'] == 'Wednesday')
                 | (df['Day_name'] == 'Thursday') | (df['Day_name'] == 'Friday')]


# # Accidents by hour (****weekdays only)
# ## Morning rush hours stand out

# In[ ]:


weekday_acc_by_hour = weekdays_df.groupby('Hour').count()['ID']
weekday_acc_by_hour.plot(kind='bar', figsize=(10, 8))


# # Accidents by hour (weekends only)
# ## A peak seems to occur 1200-1400

# In[ ]:


weekend_df = df[(df['Day_name'] == 'Saturday') | (df['Day_name'] == 'Sunday')]
weekend_acc_by_hour = weekend_df.groupby('Hour').count()['ID']
weekend_acc_by_hour.plot(kind='bar', figsize=(10, 8))


# 

# # Most accidents take place on Tuesdays
# ## Less accidents happen on weekends

# In[ ]:


accidents_day_of_week = df.groupby('Day_name').count()['ID'].sort_values(ascending=False)
accidents_day_of_week.plot(kind='bar')


# In[ ]:


accidents_day_of_week.plot(kind='pie')


# # 13th is not the most unlucky date of the month, it turns out, it's the 12th

# In[ ]:


acc_by_day_of_month = df.groupby('Day').count()['ID']
acc_by_day_of_month.plot(kind='bar', figsize=(12, 6))


# # October sees the highest number of accidents in a year

# In[ ]:


accidents_by_month = df.groupby('Month').count()['ID'].sort_values(ascending=False)
accidents_by_month.plot(kind='bar')


# In[ ]:


accidents_by_month.plot(kind='pie')


# # There has been a significant year on year increase in accidents

# In[ ]:


accidents_by_year = df.groupby('Year').count()['ID'].drop([2015, 2020])
accidents_by_year.plot(kind='bar')


# In[ ]:


plt.figure(figsize=(10, 8))
sns.scatterplot(x='Start_Lng', y='Start_Lat', hue='Severity', data=df)


# # More to follow...

# In[ ]:




