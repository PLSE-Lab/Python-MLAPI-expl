#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import calendar


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # EDA

# Let's read the dataset and save it in df variable.

# In[ ]:


df = pd.read_csv('/kaggle/input/denpasarbalihistoricalweatherdata/openweatherdata-denpasar-1990-2020v0.1.csv')


# List all the columns contained in the dataset

# In[ ]:


df.columns


# Take a peek in our dataset.

# In[ ]:


df.head(3)


# In[ ]:


df.tail(3)


# I believe our dataset were taken from the 1 January 1990 at 0.00 to 7 January 2020 at 0.00 local time.

# In[ ]:


df.shape


# Our dataset consist of 264924 instances with 29 columns.

# In[ ]:


df.info()


# Let's see if there is any missing values from our dataset.

# In[ ]:


df.isnull().sum()


# As can be seen in previous result, there are 12 columns that have a lot of missing values. Some of them have 100% of missing values. Denpasar is located in tropical country, so it wont snow in there.
# Because of that, we will drop all of that columns. Also we will drop some columns which not give meaningful information like timezone and city name (because it will have same value across all of all instance)

# In[ ]:


df.drop(['timezone', 'city_name', 'lat', 'lon','pressure', 'rain_1h',
       'rain_3h', 'rain_6h', 'rain_12h', 'rain_24h', 'rain_today', 'snow_1h',
       'snow_3h', 'snow_6h', 'snow_12h', 'snow_24h', 'snow_today',
       'clouds_all', 'weather_id','weather_icon'], axis=1, inplace=True)


# Fortunately, our dataset have timestamp. We will extract some new columns/features from the timestamp, which are year, month, day, weekday name and hour.

# In[ ]:


df['date'] = pd.to_datetime(df['dt_iso'], infer_datetime_format=True)


# In[ ]:


df.info()


# In[ ]:


df.drop(['dt_iso'], axis=1, inplace=True)


# In[ ]:


df = df.set_index('date')


# In[ ]:


df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['weekday_name'] = df.index.day_name()
df['hour'] = df.index.hour


# The dataset have the measurement in 2020, but it's only have 7 day data from 2020 (well, obviously). Because there are so little data in 2020, we will remove instances in with year of 2020.

# In[ ]:


plt.style.use('seaborn-whitegrid')


# In[ ]:


month_list = calendar.month_name[1:13]
month_list


# In[ ]:


df['1990':'2000'].groupby(['year','month'])['temp'].mean().unstack(0).plot(figsize=(16,10)).legend(bbox_to_anchor=(1,0.5))
plt.ylabel('Average Temp in C')
plt.xticks(ticks=df['month'].unique() ,labels = month_list)
plt.show()


# In[ ]:


def monthly_measurement(year, month, measurement):
    pvt_tbl = df['{}-{}'.format(year,month)].pivot_table(index='day', columns='hour', values=str(measurement)) #create pivot table
    fig, ax = plt.subplots(figsize=(12,10))
    
    month_name = calendar.month_name[month]
    
    sns.heatmap(pvt_tbl, cmap='coolwarm', ax=ax,linewidths=0.2)
    ax.set_title('Temperature in {} {}'.format(month_name, year))
    plt.show()


# In[ ]:


monthly_measurement(2019, 1, 'temp')


# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
sns.boxplot(x='month', y='temp', data=df['1990'], ax=ax)
ax.set_title('Box')
plt.show()


# In[ ]:


def percentage_of_monthly_rain(year, weather):
    
    temp = df[str(year)].groupby(['year', 'month','weather_main']).size().unstack(fill_value=0)
    total = np.sum(temp.iloc[:,:].values, axis=1)
    return ((temp.iloc[:,-1]+temp.iloc[:,-2])/total * 100).values


# In[ ]:


plt.figure(figsize=(16,8))
year = [1990, 1995, 2000, 2005, 2010, 2015]
for i in (year):
    num = percentage_of_monthly_rain(i, 'RAIN')
    plt.plot(month_list, num, label=str(i))
plt.xlabel('Month')
plt.ylabel('% Monthly Rain Occurrence')
plt.legend(bbox_to_anchor=(1,0.7))
plt.show()


# In[ ]:




