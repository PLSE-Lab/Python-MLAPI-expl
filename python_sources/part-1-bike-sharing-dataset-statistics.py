#!/usr/bin/env python
# coding: utf-8

# ## Short intro ##
# Inspired by [Tudor Lapusan's kernel](https://www.kaggle.com/tlapusan/titanic-decision-tree-implemented-as-lean-startup), I decided to do something similar but for another dataset which I found sometime ago. The purpose is to learn more about machine learning algorithms and how to handle numerical datasets. I want to start with small steps, with the simplest use cases. 
# 
# In this first kernel, I tried to extract as much knowledge as possible using just the dataset, while learning how to use Pandas, Seaborn and Matplotlib packages. Since is a small dataset, I think I've reached its limit, but if you have suggestions or ideas, feel free to comment. 

# First, import the necessary packages and the bike sharing data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# ## General statistics ##
# Examine the dataset using Pandas.

# In[ ]:


print(train_data.columns)
print('Train data: # rows: {}, # cols: {}'.format(train_data.shape[0], train_data.shape[1]))
print('Test data: # rows: {}, # cols: {}'.format(test_data.shape[0], test_data.shape[1]))

train_data.head()


# Check also for missing data.

# In[ ]:


train_data.isna().sum()


# Convert the 'datetime' column which has values of type object to values of type datetime, for easier processing.

# In[ ]:


train_data['datetime'] = pd.to_datetime(train_data['datetime'])


# In[ ]:





# ## Time series visualization  ##
# 
# Plot how many people rent bikes over the course of a day, for all seasons, but first let's split the column **datetime** into 2 columns, labeled **date** and **time**

# In[ ]:


temp = pd.DatetimeIndex(train_data['datetime'])
train_data['date'] = temp.date
train_data['hour'] = temp.time
del train_data['datetime']
train_data.head()


# Now, group the dataset by **season** and **hour**. For each season and each hour of the day, compute the mean value of the **count** column. So, in the end for each season, there will be 24 mean values. Plot them on the same graph.

# In[ ]:


gb_season = train_data.groupby(['season', 'hour'])
season_summary = gb_season.size().to_frame(name='season_summary')
season_summary = (season_summary.join(gb_season.agg( {'count': 'mean'}).rename(columns={'count': 'count_mean'}))
#                                 .join(gb_season.agg( {'count': 'max'}).rename(columns={'count': 'count_max'}))
                                .reset_index())

plt.figure(figsize=(10, 5))

# 1-spring, 2-summer, 3-fall, 4-winter
for i in season_summary.groupby('season').groups.keys():
    plt.plot(season_summary['hour'][season_summary.groupby('season').groups[i]],
            season_summary['count_mean'][season_summary.groupby('season').groups[i]])
    
plt.grid()
plt.legend(title='Season',labels=['spring', 'summer', 'fall', 'winter'])
plt.xlabel('Hour')
plt.ylabel('Mean count')
plt.title('Mean count values hourly for each season')


# From the graph above, we can see that  more people rented bikes in fall, then any other season. Also, notice that the maxium of users is around 16:30-17:30.  
# We can also plot the casual and registered users.

# In[ ]:


season_summary1 = gb_season.size().to_frame(name='season_summary1')
season_summary1 = (season_summary1.join(gb_season.agg( {'casual': 'mean'}).rename(columns={'casual': 'casual_mean'}))
#                                     .join(gb_season.agg( {'casual': 'max'}).rename(columns={'casual': 'casual_max'}))
                                    .reset_index())

season_summary2 = gb_season.size().to_frame(name='season_summary2')
season_summary2 = (season_summary2.join(gb_season.agg( {'registered': 'mean'}).rename(columns={'registered': 'registered_mean'}))
#                                     .join(gb_season.agg( {'registered': 'max'}).rename(columns={'registered': 'registered_max'}))
                                    .reset_index())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

# 1-spring, 2-summer, 3-fall, 4-winter
for i in season_summary1.groupby('season').groups.keys():
    axes[0].plot(season_summary1['hour'][season_summary1.groupby('season').groups[i]],
            season_summary1['casual_mean'][season_summary1.groupby('season').groups[i]])

axes[0].grid()
axes[0].legend(title='Season',labels=['spring', 'summer', 'fall', 'winter'])
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Mean casual users')

# 1-spring, 2-summer, 3-fall, 4-winter
for i in season_summary2.groupby('season').groups.keys():
    axes[1].plot(season_summary2['hour'][season_summary2.groupby('season').groups[i]],
            season_summary2['registered_mean'][season_summary2.groupby('season').groups[i]])
    
axes[1].grid()
axes[1].legend(title='Season',labels=['spring', 'summer', 'fall', 'winter'])
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Mean registered users')

plt.suptitle('Mean casual and registered no of users hourly for each season', multialignment='center')


# It seems that registered users rent bikes in the morning too, as opposed to the casual ones. Perhaps, to comute to work. We can see two spikes, in the morning and in the afternoon, for all season.
# 
# Let's see now how the weather influences the number of users.

# In[ ]:


gb_weather = train_data.groupby(['weather', 'hour'])
weather_summary = gb_weather.size().to_frame(name='weather_summary')
weather_summary = (weather_summary.join(gb_weather.agg( {'count': 'mean'}).rename(columns={'count': 'count_mean'}))
                                #.join(gb_weather.agg( {'count': 'max'}).rename(columns={'count': 'count_max'}))
                                .reset_index())
plt.figure(figsize=(10, 5))
for i in weather_summary.groupby('weather').groups.keys():
# 1-good weather, 2-normal, 3-bad, 4-very bad
    plt.plot(weather_summary['hour'][weather_summary.groupby('weather').groups[i]],
            weather_summary['count_mean'][weather_summary.groupby('weather').groups[i]])
plt.grid()
plt.legend(title='Weather',labels=['good', 'normal', 'bad', 'very bad'])
plt.xlabel('Hour')
plt.ylabel('Mean count')
plt.title('Mean count values hourly for each type of weather')


# From the graph above, we can observe that more people rent bikes when the weather is good or normal. Also, when the weather is very bad, no one rents a bike.

# In[ ]:


weather_summary1 = gb_weather.size().to_frame(name='weather_summary1')
weather_summary1 = (weather_summary1.join(gb_weather.agg( {'casual': 'mean'}).rename(columns={'casual': 'casual_mean'}))
#                                     .join(gb_weather.agg( {'casual': 'max'}).rename(columns={'casual': 'casual_max'}))
                                    .reset_index())

weather_summary2 = gb_weather.size().to_frame(name='weather_summary2')
weather_summary2 = (weather_summary2.join(gb_weather.agg( {'registered': 'mean'}).rename(columns={'registered': 'registered_mean'}))
#                                     .join(gb_weather.agg( {'registered': 'max'}).rename(columns={'registered': 'registered_max'}))
                                    .reset_index())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

# 1-good, 2-normal, 3-bad, 4-very bad
for i in weather_summary1.groupby('weather').groups.keys():
    axes[0].plot(weather_summary1['hour'][weather_summary1.groupby('weather').groups[i]],
            weather_summary1['casual_mean'][weather_summary1.groupby('weather').groups[i]])

axes[0].grid()
axes[0].legend(title='Weather',labels=['good', 'normal', 'bad', 'very bad'])
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Mean casual users')

# 1-good, 2-normal, 3-bad, 4-very bad
for i in weather_summary2.groupby('weather').groups.keys():
    axes[1].plot(weather_summary2['hour'][weather_summary2.groupby('weather').groups[i]],
            weather_summary2['registered_mean'][weather_summary2.groupby('weather').groups[i]])
    
axes[1].grid()
axes[1].legend(title='Weather',labels=['good', 'normal', 'bad', 'very bad'])
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Mean registered users')

plt.suptitle('Mean casual and registered no of users hourly for each type of weather', multialignment='center')


# Both casual and registered users prefer a good or normal weather to go out for a ride.
# 
# Further, we will examine the number of users for each day of the week. To accomplish this, we must find the week day based on date, where 0 represents Monday and 6 represents Sunday.

# In[1]:


# find the week day based on date.  Monday=0, Sunday=6
train_data['date'] = pd.to_datetime(train_data['date'])
train_data['weekday'] = train_data['date'].dt.weekday
train_data.head()


# In[ ]:


gb_weekday = train_data.groupby(['weekday', 'hour'])
weekday_summary = gb_weekday.size().to_frame(name='weekday_summary')
weekday_summary = (weekday_summary.join(gb_weekday.agg( {'count': 'mean'}).rename(columns={'count': 'count_mean'}))
                                #.join(gb_weather.agg( {'count': 'max'}).rename(columns={'count': 'count_max'}))
                                .reset_index())

plt.figure(figsize=(15, 7))
for i in weekday_summary.groupby('weekday').groups.keys():
    plt.plot(weekday_summary['hour'][weekday_summary.groupby('weekday').groups[i]],
            weekday_summary['count_mean'][weekday_summary.groupby('weekday').groups[i]])
plt.grid()
plt.legend(title='Weekday',labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.xlabel('Hour')
plt.ylabel('Mean count')
plt.title('Mean count values hourly for each day of the week')


# The assumption that users use the bikes to comute to work seem to hold, as there are more users during the work days than in weekends. 
# 
# Let's compare casual and registered users.

# In[ ]:


weekday_summary1 = gb_weekday.size().to_frame(name='weekday_summary1')
weekday_summary1 = (weekday_summary1.join(gb_weekday.agg( {'casual': 'mean'}).rename(columns={'casual': 'casual_mean'}))
#                                     .join(gb_weekday.agg( {'casual': 'max'}).rename(columns={'casual': 'casual_max'}))
                                    .reset_index())

weekday_summary2 = gb_weekday.size().to_frame(name='weekday_summary2')
weekday_summary2 = (weekday_summary2.join(gb_weekday.agg( {'registered': 'mean'}).rename(columns={'registered': 'registered_mean'}))
#                                     .join(gb_weekday.agg( {'registered': 'max'}).rename(columns={'registered': 'registered_max'}))
                                    .reset_index())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

for i in weekday_summary1.groupby('weekday').groups.keys():
    axes[0].plot(weekday_summary1['hour'][weekday_summary1.groupby('weekday').groups[i]],
            weekday_summary1['casual_mean'][weekday_summary1.groupby('weekday').groups[i]])

axes[0].grid()
axes[0].legend(title='Weekday',labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Mean casual users')

for i in weekday_summary2.groupby('weekday').groups.keys():
    axes[1].plot(weekday_summary2['hour'][weekday_summary2.groupby('weekday').groups[i]],
            weekday_summary2['registered_mean'][weekday_summary2.groupby('weekday').groups[i]])

axes[1].grid()
axes[1].legend(title='Weekday',labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Mean registered users')

plt.suptitle('Mean casual and registered no of users hourly for each day of the week', multialignment='center')


# The most casual users are in weekends, than during the work days as opposed to the registered users who rent bikes in the workdays. 
# 
# 

# ## Univariate visualization ##
# 
# **Quantitative features** express a count or a measurement. I'll use histogram plots.
# 
# * **Normalized feeling temperature (atemp)** The values are divided to 50 (max). As it was expected, people tend to rent bike when the temperature are around 20 Celsius.
# * **Casual users**    Number of casual users
# * **Count - Total number of users** Casual and registered
# * **Holiday** It seems people tend not to rent bike on holidays.
# * **Normalized humidity** Bigger values for humidity, more bikers.
# * **Registered users** Number of registered users
# * **Season** 1:springer, 2:summer, 3:fall, 4:winter. The number of users is almost constant no matter the season.
# * **Normalized temperature (temp)** The values are divided to 41 (max). The intuition is the same as for the feeling temperature.
# * **Weather** Most users ride when the weather is clear, while no user rides during heavy rains or snow.
#         - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
# * **Wind speed** Normalized wind speed. The values are divided to 67 (max). Lower wind speed values, more users.
# * **Working day** If day is neither weekend nor holiday is 1, otherwise is 0. There are more users during the working days.
# 
# Information gather from  the previous graphics is similar to what we found now, using histograms. However, the former representation is more representative and easy to observe.

# In[ ]:


train_data.hist(figsize=(9, 9))


# Plot the frequency of categorical features using bar plots: **season, holiday, working day, weather. ** Bar plots are more suited for categorical variables.

# In[ ]:


_, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

sns.countplot(x='holiday', data=train_data, ax=axes[0, 0])
sns.countplot(x='workingday', data=train_data, ax=axes[0, 1])
sns.countplot(x='season', data=train_data, ax=axes[1, 0])
sns.countplot(x='weather', data=train_data, ax=axes[1, 1])


# In[ ]:


users_count = np.array([train_data['casual'].sum(), train_data['registered'].sum()])
hist, bin_edges = np.histogram(users_count)

barlist = plt.bar( hist, bin_edges[:-1], color=['orange', 'blue'])
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.xlabel('casual {:{width}} registered'.format('',width=35))


# Also, the o number of registered users is greater than the number of casual users.

# ## Multivariate visualization ##
# 
# **Feature correlations.** Plot correlations matrix.

# In[ ]:


train_corr = train_data.corr(method="spearman")
plt.figure(figsize=(10, 10))
sns.heatmap(train_corr, annot=True)


# From the plot above we can se that we have strong correlations between the features **casual** and **registered**. These values together form the **count** column, which we need to predict. It might be best to remove these features since they do not have an individual importance in predicting the target variable.

# **Scatter plots** display values of two or three numerical values in 2D, respectively 3D space. Numerical features are: **feeling temperature, temperature, humidity and wind speed**.

# In[ ]:


sns.jointplot(x='humidity', y='windspeed', data=train_data, kind='scatter');
sns.jointplot(x='humidity', y='atemp', data=train_data, kind='scatter');
sns.jointplot(x='humidity', y='temp', data=train_data, kind='scatter');
sns.jointplot(x='atemp', y='temp', data=train_data, kind='scatter');
sns.jointplot(x='windspeed', y='atemp', data=train_data, kind='scatter');
sns.jointplot(x='windspeed', y='temp', data=train_data, kind='scatter');

