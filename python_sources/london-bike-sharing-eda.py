#!/usr/bin/env python
# coding: utf-8

# # Importiing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime


# # Reading and knowing the data set

# In[ ]:


# "timestamp" - timestamp field for grouping the data
# "cnt" - the count of a new bike shares
# "t1" - real temperature in C
# "t2" - temperature in C "feels like"
# "hum" - humidity in percentage
# "wind_speed" - wind speed in km/h
# "weather_code" - category of the weather
# "is_holiday" - boolean field - 1 holiday / 0 non holiday
# "is_weekend" - boolean field - 1 if the day is weekend
# "season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.

# "weather_code" category description:
# 1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity
# 2 = scattered clouds / few clouds
# 3 = Broken clouds
# 4 = Cloudy
# 7 = Rain/ light Rain shower/ Light rain
# 10 = rain with thunderstorm
# 26 = snowfall
# 94 = Freezing Fog


# In[ ]:


df = pd.read_csv("../input/london-bike-sharing-dataset/london_merged.csv")
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.columns


# # EDA

# In[ ]:


# plot heatmap with numeric features
plt.figure(figsize=(16,5))
sns.heatmap(data=df.corr(), cmap='YlGnBu', annot=True)
plt.show()


# In[ ]:


# Format and make date, hour, weekday name, weekday number, month features

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['weekday_name'] = df['timestamp'].dt.weekday_name
df['weekday'] = df['timestamp'].dt.weekday
df['month'] = df['timestamp'].dt.month


# In[ ]:


df.head()


# In[ ]:


# Bikeshares by Time of Day

data_hour = df.loc[:, ['hour', 'cnt']]
data_hour_mean = data_hour.groupby('hour').mean()['cnt'].round()

# Plot values calculated above
plt.figure(figsize=(16,5))
plt.bar(data_hour_mean.index, data_hour_mean, color='lightgreen')
plt.xlabel("Hour of Day")
plt.ylabel("Average Number of BikeShares")
plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22])
plt.title("Bikeshares by Time of Day")


# In[ ]:


# Bikeshares by Day of Week

data_weekday = df.loc[:, ['weekday_name', 'cnt']]
data_weekday_mean = data_weekday.groupby('weekday_name').mean()['cnt'].round()

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
data_weekday_mean = data_weekday_mean.reindex(index = day_order)

plt.figure(figsize=(10,5))
plt.bar(data_weekday_mean.index, data_weekday_mean)
plt.xlabel("Day of Week")
plt.ylabel("Average Number of BikeShares")
plt.title("Bikeshares by Day of Week")


# In[ ]:


# Plot weekdays vs weekends
# "is_weekend" - boolean field - 1 if the day is weekend

df['is_weekend'] = df['is_weekend'].map({0: "weekday", 1:"Weekend"})

df_weekdays_vs_weekends =df.groupby(['is_weekend']).mean()['cnt']

df_weekdays_vs_weekends.plot(kind='bar')
plt.xlabel("weekdays")
plt.ylabel("Avg. Number of BikeShares ")
plt.title("Bikeshares by weekdays")


# In[ ]:


# Plot holidays vs working days
# "is_holiday" - boolean field - 1 holiday / 0 non holiday

df['is_holiday'] = df['is_holiday'].map({0: "non holiday", 1:"holiday"})

df_hoiday =df.groupby(['is_holiday']).mean()['cnt']
df_hoiday.plot(kind='bar')
plt.xlabel("holidays")
plt.ylabel("Avg. Number of BikeShares ")
plt.title("Bikeshares by holidays")


# In[ ]:


# Plot season vs Avg. Number of BikeShares
# "season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.

df['season_new'] = df['season'].map({0:"spring", 1:"summer", 2:"fall", 3:"winter"})

df_season =df.groupby(['season_new']).mean()['cnt']

df_season.plot(kind='bar')
plt.xlabel("season")
plt.ylabel("Avg. Number of BikeShares ")
plt.title("Bikeshares by season")


# In[ ]:


df_date = df.groupby(['date']).sum()['cnt']
plt.figure(figsize=(16,5))
df_date.plot()
plt.xlabel("Date")
plt.ylabel("Number of BikeShares ")
plt.title("Bikeshares by Date")


# In[ ]:


df_month = df.groupby(['month']).mean()['cnt'].round()

plt.figure(figsize=(16,5))
plt.bar(df_month.index, df_month)
plt.xlabel("Month")
plt.ylabel("Avg. Number of BikeShares ")
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.suptitle("Bikeshares by Month")


# In[ ]:


#Bikeshares by Weather condition

# "weather_code" category description:
# 1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity
# 2 = scattered clouds / few clouds
# 3 = Broken clouds
# 4 = Cloudy
# 7 = Rain/ light Rain shower/ Light rain
# 10 = rain with thunderstorm
# 26 = snowfall
# 94 = Freezing Fog

df['weather_code'] = df['weather_code'].map({1:"Clear", 2:"few clouds", 3:"Broken clouds", 4:"Cloudy", 
                                             7:"light Rain shower", 10:"rain with thunderstorm", 26:"snowfall", 94:"Freezing Fog"})

df_weather = df.groupby(['weather_code']).mean()['cnt'].round()

plt.figure(figsize=(10,5))
plt.bar(df_weather.index, df_weather)
plt.xlabel("Weather Condition")
plt.ylabel("Avg. Number of BikeShares ")
plt.xticks(rotation=45,ha='right')
plt.title("Bikeshares by Weather condition")


# In[ ]:


#Bikeshares by weekday and day hours

df_day_vs_hour = df.loc[:,['weekday_name','hour','cnt']]
df_day_vs_hour_mean = pd.pivot_table(df_day_vs_hour,index=["weekday_name"],values=["cnt"], columns=["hour"])

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df_day_vs_hour_mean = df_day_vs_hour_mean.reindex(index = day_order)

plt.figure(figsize=(25,5))
g = sns.heatmap(data=df_day_vs_hour_mean, cmap="BuGn", annot=True, linewidths=.5, fmt=".1f" )
plt.xlabel("Day Hours")
plt.ylabel("Avg. Number of BikeShares ")
plt.xticks(rotation=45,ha='right')
plt.title("Bikeshares by weekday and day hours")


# In[ ]:


# Plot season vs day of week
# "season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.

df_season_vs_hour = df.loc[:,['season','weekday_name','cnt']]

df_season_vs_hour["season"] = df_season_vs_hour["season"].map({0: "spring", 1: "summer", 2: "fall",3: "winter"})

df_season_vs_hour_mean = pd.pivot_table(df_season_vs_hour,index=["weekday_name"],values=["cnt"], columns=["season"], aggfunc="mean")

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df_season_vs_hour_mean = df_season_vs_hour_mean.reindex(index = day_order)

plt.figure(figsize=(10,5))
g = sns.heatmap(data=df_season_vs_hour_mean, cmap="BuGn", linewidths=.5, annot=True, fmt=".1f" )
plt.xlabel("weekday_name")
plt.ylabel("Avg. Number of BikeShares ")
plt.title("Bikeshares by Season and day of week")

