#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # used to visualise data
import seaborn as sns           # used to visualise data

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # London Bike Sharing Dataset
# ## A dataset from Kaggle
# 
# This dataset can be downloaded directly from Kaggle [here](https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset). 
# 
# ### Summary of Dataset
# As a summary, this data provides historical data of London bike sharing. This data can be used to help predict future bike sharing in London City. 
# 
# I have read in all the libraries I will use in the set up code chunk above. 

# In[ ]:


# Read in raw_csv and visulaise first 5 rows of data
raw_data = pd.read_csv("../input/london-bike-sharing-dataset/london_merged.csv")
raw_data.head(5)


# In[ ]:


raw_data.dtypes


# We can see from the header of the data that we have 10 columns of data in this dataset. They all relate to the following information: 
# - *timestamp* - is the date and time of day, used for grouping the data. 
# - *cnt* - This is the cound of new bike shares.
# - *t1* - The real temperature in C.
# - *t2* - The feels like temperature in C. 
# - *hum* - The humidity as a percentage.
# - *wind_speed* - The wind speed in km/h.
# - *weather_code* - The category of the weather.
# - *is_holiday* - A boolean field representing holiday (1) or non-holiday (0).
# - *is_weekend* - A boolean field representing the weekend (1) or a weekday (0).
# - *season* - Categorical field representing the season:
#     - Spring - 0
#     - Summer - 1
#     - Fall - 2
#     - Winter - 3
# 
# Weather codes relate to:
# - 1 = Clear or mostly clear. 
# - 2 = Scattered Clouds or Few Clouds. 
# - 3 = Broken Clouds.
# - 4 = Cloudy. 
# - 7 = Rain or Light Rain.
# - 10 = Rain with Thunderstorms.
# - 26 = Snowfall. 
# - 94 = Freezing Fog.
# 
# 
# Now we have the housekeeping out of the way, let's get in to actually exploring this data. 
# 
# 
# ***
# 
# # 1. Bikeshares by Time of Day or Day of Week
# 
# The first thing we will explore is when bike shares most commonly occur as a time of day. 
# So let's do a little data manipulation and then visualise this. 

# In[ ]:


# First call raw_data and change to different variable name.
data_time = raw_data

# We will then convert our timestamp to a datetime value
data_time['timestamp'] = pd.to_datetime(data_time['timestamp'])

# Add two new columns for time of day and day of week. 
data_time['time'] = data_time['timestamp']
data_time['day'] = data_time['timestamp']

# View data to see columns were added
data_time.head(5)


# In[ ]:


# We can now check our dtypes again
data_time.dtypes


# Perfect, so now that we have two new columns, which are in datetime format, let's convert their timestamps in to a time and day value for grouping. 
# 
# ***

# In[ ]:


# Import datetime library for conversion of timestamp
import datetime

# Convert time value
data_time['time'] = data_time['time'].dt.hour

# Convert day value
data_time['day'] = data_time['day'].dt.weekday_name

data_time.head(5)


# Great, so now we have converted our values, let's plot the count of bikeshares throughout the day. 
# 
# ***
# 
# # Data Visulisation

# In[ ]:


# Create groupby function for time of day
data_time_time = data_time.groupby('time').mean()

# Plot values calculated above
plt.figure()
plt.bar(data_time_time.index, data_time_time['cnt'])
plt.xlabel("Hour of Day")
plt.ylabel("Average Number of BikeShares")
plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22])
plt.suptitle("Bikeshares by Time of Day")
plt.show()


# From this we can see that the early hours of the day have minimal usage. Which obviously makes sense based on work hours and so on. We also have a couple of major peaks throughout the day, at around 8am and again at 5-6pm. These are times when people are starting or finishing work, so again makes sense. So let's now look at the days of the week. 
# 
# ***

# In[ ]:


# Create groupby function for the day of the week
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
data_time_day = data_time.groupby('day').mean()
data_time_day = data_time_day.reindex(index = day_order)

# Plot values calculated above
plt.figure()
plt.bar(data_time_day.index, data_time_day['cnt'])
plt.xlabel("Day of Week")
plt.ylabel("Average Number of BikeShares")
plt.suptitle("Bikeshares by Day of the Week")
plt.xticks(rotation=90)
plt.ylim([900, 1300])
plt.show()


# We can see that Monday through Friday is when the majority of bikeshares are taken. Therefore, this must be a use of transport for those that are working or going to university etc. Saturday and Sunday usage drops below 1000, indicating that people either stay home or opt for different transportation methods.
# 
# ***
# 
# Let's have a look and see if the number of shares change throughout the day in relation to some of our other information in the dataset.

# In[ ]:


# Create a plot with 5 axes.
fig,(ax1, ax2, ax3, ax4, ax5)= plt.subplots(nrows=5)
fig.set_size_inches(18,25)

# Create all the subplots
sns.pointplot(data=data_time, x='time', y='cnt', ax=ax1)
sns.pointplot(data=data_time, x='time', y='cnt', hue='is_holiday', ax=ax2)
sns.pointplot(data=data_time, x='time', y='cnt', hue='is_weekend', ax=ax3)
sns.pointplot(data=data_time, x='time', y='cnt', hue='season', ax=ax4)
sns.pointplot(data=data_time, x='time', y='cnt', hue='weather_code',ax=ax5)


# We can see that holidays and weekends follow similar trends, with a much more distinct curve that is consistant throughout the day. Whilst, weekdays have to distinctive spikes around the time of commuters beginning work for a day. 
# The seasons have a similar trend, with slightly different absolute values. Winter in particular, indicates a big drop in the amount of bikeshares taken through the day. 
# 
# ***
# 
# # Correlation Matrix
# 
# Now, what things are most related to eachother in this dataset?  
# We can create a correlation matrix to take a look. 

# In[ ]:


# Create a correlation matrix
corrmat = data_time.corr()
f, ax = plt.subplots(figsize = (10,10))
sns.heatmap(corrmat, vmax=1, annot=True);


# From the above matrix, we can see that humidity has the strongest relationship, although negative with the number of bikeshares. Time and temperatures also have appear to have an impact on the number of bikeshares, but to a smaller degree. 
# 
# None of these correlations are really that strong though and so inclusion of most of these values would help us to provide a much stronger prediction.
# But let's leave the prediction for another day! 
