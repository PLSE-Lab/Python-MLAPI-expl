#!/usr/bin/env python
# coding: utf-8

# This kernel provides predictive analytics on the Bike Sharing dataset. 
# 
# Dataset info:
# 
# "**timestamp**" - timestamp field for grouping the data
# 
# "**cnt**" - the count of a new bike shares
# 
# "**t1**" - real temperature in C
# 
# "**t2**" - temperature in C "feels like"
# 
# "**hum**" - humidity in percentage
# 
# "**windspeed**" - wind speed in km/h
# 
# "**weathercode**" - category of the weather
# 
# "**isholiday**" - boolean field - 1 holiday / 0 non holiday
# 
# "**isweekend**" - boolean field - 1 if the day is weekend
# 
# "**season**" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.
# 
# 
# 
# "weather_code" category description:
# 1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity 
# 2 = scattered clouds / few clouds 
# 3 = Broken clouds 
# 4 = Cloudy 
# 7 = Rain/ light Rain shower/ Light rain 
# 10 = rain with thunderstorm 
# 26 = snowfall 
# 94 = Freezing Fog

# Import necessary packages

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


# Load the dataset

# In[ ]:


df = pd.read_csv("/kaggle/input/london-bike-sharing-dataset/london_merged.csv")
df.head()


# Convert timestamp column to datetime object

# In[ ]:


df.timestamp = pd.to_datetime(df.timestamp)
df.timestamp[0].month


# In[ ]:


df.info()


# There are over 17K samples, no missing data, summary stats of the numeric variables:

# In[ ]:


df.describe()


# Time difference between observations are **1 hour**, in average

# In[ ]:


df.timestamp.diff().mean()


# It covers a two year period between 4/1/2015 and 3/1/2017

# In[ ]:


df.timestamp.describe()


# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(df.cnt,label="Bike Share Distribution")
mean_share = df.cnt.mean()
plt.plot([mean_share,mean_share],[0,0.0012],"-",linewidth=5,label="Average # of Bike Share")
plt.text(mean_share+100,0.001201,int(mean_share),fontweight="bold",fontsize=12)
plt.legend()


# Lets plot monthly counts of bike shares within 2 years period. Summer shares are significantly higher as compared to winter as expected

# In[ ]:


df.groupby(pd.Grouper(key="timestamp",freq="1M")).sum()["cnt"].plot()


# Let's now visualise the the correlation between the parameters using Seaborn's pairplot. The diagonal line shows each parameter's distribution in a histogram chart while the rest visualise correlations in scatter chart. For speeding up the calculations and plotting as well as a better correlation visibility, we will use 5% of the randomly sampled chunk of the data for pairplot. Some of the initial findings are:
# 
# * As expected, the real and felt temperature are highly correlated
# * Negative correlation between temperature and humidity
# * Positive correlation between temperature and bike share count
# * Negative correlation between humidity and bike share count

# In[ ]:


sns.pairplot(df.sample(frac=0.05))


# Calculate correlation (absolute) between parameters and display with a heatmap. Since we will think number of bike share as our target, let's see which parameters are correlated more

# In[ ]:


corrs = abs(df.corr())

fig = plt.figure(figsize=(10,8))
sns.heatmap(corrs,annot=True)


# Let's make a look up table for weather codes and visualise the distribution of weather types:
# 
# * Weather is 51% of  cloudy (scattered, broken or just cloudy)
# * 1/3 of a time it is clear sky
# * Almost no days with snow or thunderstorm
# 

# In[ ]:


weather_lookup = {
1 : "Clear",
2 : "Scattered clouds",
3 : "Broken clouds",
4 : "Cloudy",
7 : "Rain",
10 : "Rain with thunderstorm",
26 : "Snowfall",
94 : "Freezing Fog"}

weather_counts = df.weather_code.value_counts()
weather_counts.index = [weather_lookup[i] for i in weather_counts.index]
weather_counts.plot(kind="pie",autopct="%.0f%%")
plt.ylabel("")
plt.title("Weather Type Distribution",fontweight="bold")


# ## Feature & Target relationships
# 
# 
# > From this point on the variable which gives the number of bike shares per hour (*cnt*) will be our **target** variable.
# 
# Let's see how is the relationship between the target and other features. Maybe we can generate some new features using the columns. People tend to share bikes more on the non-holiday period. However note that population size difference between the two groups is enormous.

# In[ ]:


plt.axhline(y=mean_share,linewidth=5,c="deepskyblue",label="Mean")
sns.barplot(x="is_holiday",y="cnt",data=df)
plt.legend()


# Let's make 10 weather temperature groups and calculate each group's average bike sharing counts:
# 
# * When temperature increases, bike sharing increases (positive correlation)
# * Bike sharing increases drammatically when temperature is above 17 degrees
# * No difference was observed when t1 is changed with t2

# In[ ]:


temps = pd.qcut(df.t1,10)
sns.barplot(y=temps,x="cnt",data=df,orient="h")
plt.axvline(x=mean_share,label="Mean",c="deepskyblue")
plt.legend()


# Similarly, humidity values are gathered into 10 equally sized groups, and group relation with the target value is visualised:
# 
# > Humidity is **negatively** correlated with bike share

# In[ ]:


hums = pd.qcut(df.hum,10)
sns.barplot(y=hums,x="cnt",data=df,orient="h")
plt.axvline(x=mean_share,c="deepskyblue",label="Mean")
plt.legend()


# And, wind speeds are gathered into 10 closely sized groups, and group relation with the target value is visualised:
# 
# > Even though there was no correlation appeared in the pairplot, surprisingly wind speed seems to be **positively** correlated with the target variable

# In[ ]:


winds = pd.qcut(df.wind_speed,10)
sns.barplot(y=winds,x="cnt",data=df,orient="h")
plt.axvline(x=mean_share,c="deepskyblue",label="Mean")
plt.legend()


# As the weather code gets rainier and colder, the number of bike shares tend to decline

# In[ ]:


#first replace values with the categorical names in the weather_code column
df.weather_code.replace({
1 : "Clear",
2 : "Scattered clouds",
3 : "Broken clouds",
4 : "Cloudy",
7 : "Rain",
10 : "Rain with thunderstorm",
26 : "Snowfall",
94 : "Freezing Fog"},inplace=True)

sns.barplot(y="weather_code",x="cnt",data=df,orient="h",
            order=["Clear","Scattered clouds","Broken clouds","Cloudy","Rain","Rain with thunderstorm","Snowfall","Freezing Fog"])
plt.axvline(x=mean_share,c="deepskyblue",label="Mean")
plt.legend()


# People tend to share bikes on the commuting, working days more than the holidays

# In[ ]:


df.is_holiday.replace({1:"Holiday",0:"Workday"},inplace=True)
sns.barplot(x="cnt",y="is_holiday",data=df,orient="h")
plt.axvline(x=mean_share,c="deepskyblue",label="Mean")
plt.legend()


# Similarly, people tend to use bike sharing during the weekdays

# In[ ]:


df.is_weekend.replace({1:"Weekend",0:"Weekday"},inplace=True)
sns.barplot(x="cnt",y="is_weekend",data=df,orient="h")
plt.axvline(x=mean_share,c="deepskyblue",label="Mean")
plt.legend()


# In term of seasons; summers have the highest number of bike sharing, and sharing numbers decline as it gets colder 

# In[ ]:


df.season.replace({0:"Spring",1:"Summer",2:"Fall",3:"Winter"},inplace=True)
sns.barplot(y="season",x="cnt",data=df,orient="h",order=["Spring","Summer","Fall","Winter"])

plt.axvline(x=mean_share,c="deepskyblue",label="Mean")
plt.legend()


# ### Predictive Analytics
# 
# XGBosst regression codes will be added here in the future commits
