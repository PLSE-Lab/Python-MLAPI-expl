#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

df = pd.read_csv("../input/metro-bike-share-trip-data.csv", low_memory=False)


# Lets start by looking at the statistics of the trip durations

# In[ ]:



format = "%Y-%m-%dT%H:%M:%S"
start_time = pd.to_datetime(df["Start Time"], format=format)
end_time = pd.to_datetime(df["End Time"], format=format)

duration = pd.DatetimeIndex(end_time - start_time)
duration = pd.DataFrame(duration.hour*60 + duration.minute)


# In[ ]:


print(duration.describe())


# In[ ]:


print(df.Duration.describe())


# From the data processing section on https://bikeshare.metro.net/about/data/ we find that all trips shorter than 1 minute and longer than 24 hours are removed before publication. Looking at the duration data from the calculation from the start and end times, we find that the fastest trip is 1 minute and the longest trip is 1440 minutes, or 24 hours. The documentation states that the duration column is the trip length in minutes, however this is clearly not the case. Instead it looks like the columns is in seconds. We replace the duration column data with the actual calculated duration times in mintes.

# In[ ]:


df.Duration = duration


# In[ ]:


plt.figure(figsize=(16, 8))
df.Duration.hist(bins=30, range=(0, 60))
plt.xlabel("Minutes")
plt.ylabel("Count")


# The Median trip is 10 minutes and the average trip duration is 24 minutes.

# Lets now look at how the different plans are distributed amoung the passholders.

# In[ ]:


for days, count in Counter(df["Plan Duration"].fillna("nan")).items():
    print("days = {}, count = {}, percentage = {}%".format(days, count, int(100*count/len(df))))


# Here we can see that there are 3 different plans. 1 day plan, 30 day plan and a 1 year plan. The 30 day plan is the most common with 61% of all the passholders, while the  1 year plan is the least common with only 7%.

# In[ ]:


bikes = df["Bike ID"].dropna()
print(bikes.value_counts().describe())


# In[ ]:


plt.figure(figsize=(16, 8))
bikes.value_counts().plot(kind="hist", bins=35, range=(0,300))
plt.xlabel("Count of occurrences each bike was used")


# We have 763 unique bikes and each has been used on average 173 times.

# In[ ]:


for days, count in Counter(df["Passholder Type"]).items():
    print("type = {}, count = {}, percentage = {}%".format(days, count, int(100*count/len(df))))


# There is a clear correlation between the passholder type column and the plan duration column. Obviously, the monthly pass has a duration of 30 days, i.e. 61% of the trips. The flex pass correlates with the 1 year plan and the walk-up type with the dayly plan. The Staff annual type seems to correspond to about about 50% of the nan values in the plan duration column.

# In[ ]:


for days, count in Counter(df["Trip Route Category"]).items():
    print("type = {}, count = {}, percentage = {}%".format(days, count, int(100*count/len(df))))


# In[ ]:


start_station = df["Starting Station ID"].dropna().value_counts()
end_station = df["Ending Station ID"].dropna().value_counts()

stations = pd.concat((start_station, end_station), axis=1, sort=False)
stations = stations.reset_index(drop=True)


# In[ ]:


print(len(stations))


# In[ ]:


print(stations.corr())


# In[ ]:


stations.plot(figsize=(16, 8))
plt.xlabel("Stations")
plt.ylabel("Count of each station")
plt.legend()


# From the start and end station ID columns we find that there are 67 different stations.

# In[ ]:


from math import sin, cos, sqrt, atan2, radians

lat1 = df["Starting Station Latitude"].apply(radians)
lon1 = df["Starting Station Longitude"].apply(radians)
lat2 = df["Ending Station Latitude"].apply(radians)
lon2 = df["Ending Station Longitude"].apply(radians)

dlon = lon2 - lon1
dlat = lat2 - lat1

R = 6373.0

a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

df["distance"] = R * c


# In[ ]:


df["distance"].describe()


# In[ ]:


plt.figure(figsize=(16, 8))
df.distance.hist(range=(0,4), bins=50)
plt.xlabel("km")


# Many trips seems to start and end at the same station, hence the 0km distance. The median distance bewteen two stations is 1km.

# In[ ]:


stations = df["Starting Station ID"].dropna().unique()

distance, std, count = list(), list(), list()
for s1 in stations:
    d_row, c_row, s_row = list(), list(), list()
    for s2 in stations:
        sdf = df[(df["Starting Station ID"] == s1) & (df["Ending Station ID"] == s2)]
        c_row.append(len(sdf))
        d = 0
        if len(sdf) > 0:
            s = np.std(sdf.distance)
            d = np.average(sdf.distance)
            if np.isnan(d): 
                d = 0
                s = 0
            if d > 5: d = 5
        d_row.append(d)
        s_row.append(s)
    distance.append(d_row)
    count.append(c_row)
    std.append(s_row)
    
distance = pd.DataFrame(data=distance, index=stations, columns=stations)
std = pd.DataFrame(data=std, index=stations, columns=stations)
count = pd.DataFrame(data=count, index=stations, columns=stations)


# In[ ]:


plt.figure(figsize=(16, 8))
sns.heatmap(count)
plt.title("Number of trips between stations")
plt.xlabel("Ending stations")
plt.ylabel("Starting stations")


# In[ ]:


plt.figure(figsize=(16, 8))
sns.heatmap(distance)
plt.title("Distance between stations")
plt.xlabel("Ending stations")
plt.ylabel("Starting stations")


# In[ ]:


plt.figure(figsize=(16, 8))
sns.heatmap(std)
plt.title("Std of the distance between stations")
plt.xlabel("Ending stations")
plt.ylabel("Starting stations")


# The above plot shows that we have variance in the distance on different trips between the same stations.

# In[ ]:


rows = std[std > 1].sum(axis=0)
rows = rows[rows > 0]

cols = std[std > 1].sum(axis=1)
cols = cols[cols > 0]

ambiguous_stations = np.unique(list(rows.index) + list(cols.index))
print(ambiguous_stations)


# In this list of 18 stations we have uncertanty in the location of the station.

# Now lets check if our calculated trip distance correlate with the trip route category data.

# In[ ]:


df.loc[df["Trip Route Category"] == "Round Trip", "distance"].describe()


# In[ ]:


df.loc[df["Trip Route Category"] == "One Way", "distance"].describe()


# Looks good so far. The round trip distance is always zero while the one way trips are not. However, some one way trips are zero, lets check why.

# In[ ]:


df.loc[(df["Trip Route Category"] == "One Way") & (df.distance < 1e-5)].head()


# Only one trip, good. However, looking at the start and end stations, we see that they are both 3039. Should this not have been classified as a round trip?

# Now lets investigate if passholders do different kind of trips.

# In[ ]:


year = df.loc[df["Passholder Type"] == "Flex Pass", "distance"]
month = df.loc[df["Passholder Type"] == "Monthly Pass", "distance"]
day = df.loc[df["Passholder Type"] == "Walk-up", "distance"]

data = [np.sum(year < 1e-5)/len(year), np.sum(month < 1e-5)/len(month), np.sum(day < 1e-5)/len(day)]
index = ["Flex Pass", "Monthly Pass", "Walk-up"]
zero_distance_trips = pd.DataFrame(data=data, index=index)

zero_distance_trips.plot.bar(figsize=(16, 8))


# 20% of the walk-up trips and 5% of the year and monthly pass trips are round station trips. Can we assume that the year and monthly passes are used by commuters while the daily pass is more common for tourists?

# In[ ]:


plt.figure(figsize=(16, 8))
year.plot.hist(bins=22, range=(0.1,4), alpha=0.2, density=True, label="Flex Pass")
month.plot.hist(bins=22, range=(0.1,4), alpha=0.2, density=True, label="Monthly Pass")
plt.legend()


# It looks like the distrtibution of the distance of the yearly passholder trips are shiften from the monthly. Does this mean that people with longer commute distance are more likely to buy a flex pass than a monthly pass?

# In[ ]:


time = pd.DatetimeIndex(start_time)
start_hour = pd.DataFrame(time.hour + time.minute/60)

fig, ax = plt.subplots(figsize=(16, 8))
for ph in ["Flex Pass", "Monthly Pass"]:
    d = start_hour[df["Passholder Type"] == ph]
    plt.hist(d.values, bins=100, range=(0,24),  density=True, label=ph, alpha=0.2)
    plt.xlabel("Hour")

plt.legend()

for ph in ["Walk-up"]:
    fig, ax = plt.subplots(figsize=(16, 8))
    d = start_hour[df["Passholder Type"] == ph]
    plt.hist(d.values, bins=100, range=(0,24),  density=True, label=ph, alpha=0.2)
    plt.title(ph)
    plt.xlabel("Hour")

plt.legend()


# Here we see the histogram of the start time of the trips for the three passholder types. We can see that the flex and monthly passes have three peaks during the day. Morning, lunch and early evening. The morning and evening peaks are shifted for the monthly pass relative the flex pass. For the day pass we have a smoother distributions and a higher activity during the night.

# In[ ]:


data = list()
for ph in ["Flex Pass", "Monthly Pass", "Walk-up"]:
    data.append(df.loc[df["Passholder Type"] == ph, "Starting Station ID"].dropna().value_counts())

stations = pd.concat(data, axis=1, sort=False)
station_names = list(stations.index)
stations = stations.reset_index(drop=True)
stations.columns = ["Flex Pass", "Monthly Pass", "Walk-up"]
stations = stations.apply(lambda s: s/s.sum())


# In[ ]:


stations.corr()


# The popularity of the different stations are relativelly equal bewteen the different passes. We see the highest correlation between the flex and monthly pass. 

# In[ ]:


stations.plot(figsize=(16, 8))
plt.xlabel("Stations")
plt.ylabel("Count of each station")
plt.legend()


# Many stations are very similar between the passes. Lets find the ones that differs the most between the monthly and daily pass.

# In[ ]:


diff = stations["Monthly Pass"] - stations["Walk-up"]
diff = diff.apply(abs).nlargest(5).index.map(lambda s: station_names[s])
print(list(diff))


# These are the 5 stations that differs most beteen the monthly and walk-up pass. The actual station names are:
# 
# 3069,Broadway & 3rd,7/7/2016,DTLA,Active
# 
# 3064,Grand & 7th,7/11/2016,DTLA,Active
# 
# 3030,Main & 1st,7/7/2016,DTLA,Active
# 
# 3014,Union Station West Portal,7/7/2016,DTLA,Active
# 
# 3082,Traction & Rose,8/29/2016,DTLA,Active
# 
# 
# 
# 
