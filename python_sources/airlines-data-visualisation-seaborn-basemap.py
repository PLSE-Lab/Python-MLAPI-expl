#!/usr/bin/env python
# coding: utf-8

# I am going to focus my attention only on data visualisation here. In order to make the charts as tidy as possible I am going to use line, bar and pie charts. Hopefully, after reading this kernel more advanced users will be able to enrich their knowledge, because I am going to create some charts using Matplotlib sub library called Basemap as well as Geopy - library helpful for retrieving object's longitude and latitude

# ## Data description
# 
# Data source: http://stat-computing.org/dataexpo/2009/the-data.html
# 
# For the sake of simplicty, I am going to include only two most recent years in analysis (2007 and 2008). However, it still takes a lot of Kaggle's resources, so in order to perform analysis more efficiently, some database resolutions should be introduced. 
# 
# **Dataset features description:**
# 1. Year	1987-2008
# 1. Month	1-12
# 1. DayofMonth	1-31
# 1. DayOfWeek	1 (Monday) - 7 (Sunday)
# 1. DepTime	actual departure time (local, hhmm)
# 1. CRSDepTime	scheduled departure time (local, hhmm)
# 1. ArrTime	actual arrival time (local, hhmm)
# 1. CRSArrTime	scheduled arrival time (local, hhmm)
# 1. UniqueCarrier	unique carrier code
# 1. FlightNum	flight number
# 1. TailNum	plane tail number
# 1. ActualElapsedTime	in minutes
# 1. CRSElapsedTime	in minutes
# 1. AirTime	in minutes
# 1. ArrDelay	arrival delay, in minutes
# 1. DepDelay	departure delay, in minutes
# 1. Origin	origin IATA airport code
# 1. Dest	destination IATA airport code
# 1. Distance	in miles
# 1. TaxiIn	taxi in time, in minutes
# 1. TaxiOut	taxi out time in minutes
# 1. Cancelled	was the flight cancelled?
# 1. CancellationCode	reason for cancellation (A = carrier, B = weather, C = NAS, D = security)
# 1. Diverted	1 = yes, 0 = no
# 1. CarrierDelay	in minutes
# 1. WeatherDelay	in minutes
# 1. NASDelay	in minutes
# 1. SecurityDelay	in minutes
# 1. LateAircraftDelay	in minutes
# 
# My main focus will be on Southwest Airlines, but I am going to compare those airlines with other ones at a couple of aspects as well.

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print(os.listdir("../input"))
pd.options.display.max_columns = 40


# In[ ]:


data2007 = pd.read_csv(r'../input/2007/2007.csv')
data2008 = pd.read_csv(r'../input/2008/2008.csv')
data1 = pd.concat([data2007, data2008],axis=0).reset_index(drop=True)
data1.CancellationCode = data1.CancellationCode.fillna('NoCanc')

data = data1.copy()


# In order to make actions faster, I am going to convert the format of every object type variable into category and every 64-bit variable to 32-bit. This operation decreases required memory size by a half.

# In[ ]:


print('Total memory usage before variables optimization:', data.memory_usage(deep=True).sum() * 1e-6)
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype('category')
    else:
        data[col] = data[col].astype('float32')
print('Total memory usage after variables optimization:', data.memory_usage(deep=True).sum() * 1e-6)


# Datetime variable will be useful for further analysis, so in this place I am going to use *DayofMonth*, *Month* and *CRSDepTime* variables into *Date*.

# In[ ]:


for i in ['DayofMonth', 'Month']:
    data[i] = data[i].astype(int).astype(str).map(lambda x: '0'+x if len(x)==1 else x)
    

data.CRSDepTime = data.CRSDepTime.astype(str).map(lambda x: '0'+x[:len(x)-2] if len(x[:len(x)-2])==3 else '00'+x[:len(x)-2] if len(x[:len(x)-2])==2 else '000'+x[:len(x)-2] if len(x[:len(x)-2])==1 else x[:len(x)-2])

data['Date'] = data.DayofMonth.astype(str) + data.Month.astype(str) + data.Year.astype(int).astype(str) + data.CRSDepTime.astype(str)
data['Date'] = pd.to_datetime(data.Date , format='%d%m%Y%H%M')


# ## Comparison with competition
# During comparison, I have chosen to focus only on delays and market share evaluated in the amount of flights in both years.

# In[ ]:


# Median of departure delays in 2007 and 2008 between each carrier
fig, axarr = plt.subplots(1,2,figsize=(25,7))
plt.suptitle('Median of departure delays by airlines', fontsize=15)

a = data[(data.Year==2007) & (data.DepDelay>0)][['DepDelay','UniqueCarrier']].groupby('UniqueCarrier').median()
fig1 = sns.barplot(x=a.index, y=a.DepDelay,ax=axarr[0]).grid(b=True,color='lightgrey',alpha=0.5)
axarr[0].title.set_text('2007')
axarr[0].axhline(y=np.median(data.DepDelay[data.DepDelay>0][data.Year==2007]))


b = data[(data.Year==2008) & (data.DepDelay>0)][['DepDelay','UniqueCarrier']].groupby('UniqueCarrier').median()
fig2 = sns.barplot(x=b.index, y=b.DepDelay,ax=axarr[1]).grid(b=True,color='lightgrey',alpha=0.5)
axarr[1].title.set_text('2008')
axarr[1].axhline(y=np.median(data.DepDelay[data.DepDelay>0][data.Year==2008]))


for i in range(0,2):
    axarr[i].set_ylabel('Delay (in minutes)', fontsize=11)
    axarr[i].set_xlabel('Carrier', fontsize=11)


# In[ ]:


# Median of arrival delays in 2007 and 2008 between each carrier
fig, axarr = plt.subplots(1,2,figsize=(25,7))
plt.suptitle('Median of arrival delays by airlines', fontsize=15)

a = data[(data.Year==2007) & (data.ArrDelay>0)][['ArrDelay','UniqueCarrier']].groupby('UniqueCarrier').median()
sns.barplot(x=a.index, y=a.ArrDelay,ax=axarr[0]).grid(b=True,color='lightgrey',alpha=0.5)
axarr[0].title.set_text('2007')
axarr[0].axhline(y=np.median(data.ArrDelay[data.ArrDelay>0][data.Year==2007]))



b = data[(data.Year==2008) & (data.ArrDelay>0)][['ArrDelay','UniqueCarrier']].groupby('UniqueCarrier').median()
sns.barplot(x=b.index, y=b.ArrDelay,ax=axarr[1]).grid(b=True,color='lightgrey',alpha=0.5)
axarr[1].title.set_text('2008')
axarr[1].axhline(y=np.median(data.ArrDelay[data.ArrDelay>0][data.Year==2007]))



for i in range(0,2):
    axarr[i].set_ylabel('Delay (in minutes)', fontsize=11)
    axarr[i].set_xlabel('Carrier', fontsize=11)


# Southwest is keeping its departure delay and arrival delay slightly below median value. It is worth to underline that they were not able to decrease the time of departure delay in 2008 and arrival delay slightly increased.

# In[ ]:


Market = pd.DataFrame({'Carriers':data1.UniqueCarrier.value_counts().index,'AmtOfFlights':data1.UniqueCarrier.value_counts().values}).sort_values(by='AmtOfFlights', ascending=False)
x = Market.iloc[-2:].apply(np.sum,axis=0)
x.Carriers = 'Other'
Market.iloc[-1] = x
Market = Market.drop(18, axis=0)


plt.figure(figsize=(15,10))
plt.pie(Market.AmtOfFlights, labels = Market.Carriers)
plt.title('Amount of flights by each carrier in 2007 and 2008', fontsize=15)


# I am fully aware that pie chart is usually disregarded in DS communities (I guess due to limited readability), but in my opinion, we can spot the most important information - Southwest is far away from its competitors with the twice amount of flights as their second biggest competitor - American Airlines. From the chart, we can also see, that Southwest Airlines is meeting around 20% demand for flights around the United States.

# ## Southwest Airlines basic characteristics

# One of the possibly best performance measures in provided dataset is an amount of flights by each carrier, so I decided to plot it month by month.

# In[ ]:


# Amount of flights in each month
data1['MonthT'] = np.where(data1.Year==2008,data1.Month+12,data1.Month)
data1['Ones'] = 1
FlightsEM = data1[data1.UniqueCarrier=='WN'].groupby('MonthT')['Ones'].sum()

plt.figure(figsize=(15,7))
sns.lineplot(x = FlightsEM.index, y=FlightsEM.values).grid(b=True, color='lightgrey')
plt.title('Number of flights each month',fontsize=15)
MonthNames = []
MonthNamesShort = []
for i in range(1,13):
    MonthNames.append(calendar.month_name[i])
    MonthNamesShort.append(calendar.month_abbr[i])
plt.xticks(np.arange(1,25,1), labels=MonthNamesShort*2)
plt.xlabel('Month',fontsize=11)
plt.ylabel('Amount of flights',fontsize=11)


# We are including only two years, but it seems that at least in the case of Southwest Airlines, we are dealing with seasonality with the lowest amount of flights around February and highest around July or August. 
# 
# Also, because of seasonal character of data, it is required to compare any changes to the adequate month in previous years.

# In[ ]:


# Month to month total flights comparison
Mo2007Flights = data.Month[(data.UniqueCarrier=='WN') & (data.Year==2007)].value_counts()
Mo2008Flights = data.Month[(data.UniqueCarrier=='WN') & (data.Year==2008)].value_counts()

m2m = ((Mo2008Flights - Mo2007Flights) / Mo2007Flights) * 100
plt.figure(figsize=(15,7))
sns.barplot(x=m2m.index, y=m2m.values)
plt.title('Total flights growth (month to month comparison)', fontsize=15)
plt.xticks(range(0,12),labels=MonthNames)
plt.yticks(range(-2,10,2),labels=['-2%','0%','2%','4%','6%','8%'])
plt.xlabel('Month',fontsize=11)
plt.ylabel('Amount of flights',fontsize=11)
plt.axhline(0,color='green')


# While dealing with seasonal data, it would be unfair to consider flight grow comparing previous month to recent - that is why I decided to compare each month from 2008 to adequate month in 2007. As you may see, top performance in comparison was reached in February and after that, month to month flights grow was on average around 3% percent. After August things went downhill, which may be considered as the "peak" of financial crisis.
# 
# Seasonality can be spotted not only if it comes to amount of flights, but delays as well.

# In[ ]:


# Departure delay in each weekday
fig, axarr = plt.subplots(1,2,figsize=(20,7))

WeekdayName = []
for i in range(0,7):
    WeekdayName.append(calendar.day_name[i])

plt.suptitle('Average delay by day of a week',fontsize=15)
    
dataWN = data[data.UniqueCarrier=='WN']
delay1 = dataWN.groupby('DayOfWeek')['DepDelay'].mean()
sns.barplot(x=delay1.index, y=delay1.values, ax=axarr[0]).yaxis.grid(True, color='lightgrey',alpha=0.5)
axarr[0].set_title('Departure delay', fontsize=11)


delay1 = dataWN.groupby('DayOfWeek')['ArrDelay'].mean()
sns.barplot(x=delay1.index, y=delay1.values, ax=axarr[1]).yaxis.grid(True, color='lightgrey',alpha=0.5)
axarr[1].set_title('Arrival delay',fontsize=11)

for i in [0,1]:
    axarr[i].set_xticklabels(WeekdayName)
    axarr[i].set_xlabel('Day', fontsize=11)
    axarr[i].set_ylabel('Delay (in minutes)', fontsize=11)
    axarr[i].set_xticklabels(WeekdayName)


# In[ ]:


# Delays in each month
fig, axarr = plt.subplots(1,2,figsize=(20,7))

dataWN = data[data.UniqueCarrier=='WN']
delay1 = dataWN.groupby('Month')['DepDelay'].mean()
sns.barplot(x=delay1.index, y=delay1.values, ax=axarr[0]).yaxis.grid(True, color='lightgrey',alpha=0.5)

dataWN = data[data.UniqueCarrier=='WN']
delay1 = dataWN.groupby('Month')['ArrDelay'].mean()
sns.barplot(x=delay1.index, y=delay1.values, ax=axarr[1]).yaxis.grid(True, color='lightgrey',alpha=0.5)


# No matter if we are talking about departure or arrival delays, the hardest day for Southwest Airlines is Friday, and the easiest is Saturday. One of the things which you can spot from charts so far is that Southwest is able to catch up delays and land on a destination airport earlier than we could expect from departure delay.

# In[ ]:


# Percent of delayed flights
fig, axarr = plt.subplots(2,1,figsize=(15,10))
sns.kdeplot(data1.DepDelay[data1.UniqueCarrier=='WN'], legend=False, bw=5, ax=axarr[0]).grid(b=True,color='lightgrey')
axarr[0].title.set_text('Distribution of departure delays')
axarr[0].set_xlim([0,150])
axarr[0].set_yticklabels(['0%','1%','2%','3%','4%'])


sns.kdeplot(data1.ArrDelay[data1.UniqueCarrier=='WN'], legend=False, bw=5, ax=axarr[1]).grid(b=True,color='lightgrey')
plt.xlabel('Delay (in minutes)',fontsize=11)
plt.ylabel('% of delays', fontsize=11)
axarr[1].title.set_text('Distribution of arrival delays')
axarr[1].set_yticklabels(['0%','1%','2%','3%','4%', '5%', '6%'])
axarr[1].set_xlim([0,150])


# In[ ]:


import datetime
fig, axarr = plt.subplots(figsize=(15,7))

dataWN['Hour'] = dataWN.Date.map(lambda x: x.hour)
DelaysByHour = dataWN.groupby('Hour')[['ArrDelay','DepDelay']].mean()
plt.plot( DelaysByHour.index, DelaysByHour.ArrDelay, color='green', linewidth=4)
plt.plot( DelaysByHour.index, DelaysByHour.DepDelay, color='red', linewidth=4)
plt.legend(labels = ['Arrival delay', 'Departure delay'])
plt.grid(b='True', color='lightgrey')
plt.title('Average arrival and departure delay (by hours)', fontsize=15)
plt.xlabel('Hour', fontsize=11)
plt.ylabel('Delay (in minutes)', fontsize=11)
plt.xticks(range(5,25,2))


# Both arrival and departure delay are increasing with day duration, peaking just before ending air traffic.
# 
# ## Traffic map
# Now, I am going to present most often flight paths took by Southwest Airlines. In order to do it, I will use two separate libraries - Basemap and Geopy. Using Basemap allows to create a map of a specific area, add lines representing each flight patch, choose color scheme, etc. Geopy gives the opportunity to automatically search for each airport without decoding their IATA codes and searching for longitude and latitude.
# 
# Red lines represent flight patches took most often by Southwest Airlines - no matter if direction is from X to Y or Y to X. The colder the color is, the more rare specific flight patches are.

# In[ ]:


names = ['id','name','city','country','iata','icao','lat','lon','alt','timezone','dst','tz','type','source']
airports = pd.read_csv('https://github.com/ipython-books/cookbook-2nd-data/blob/master/airports.dat?raw=true', names=names)


# In[ ]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

for carrier in data.UniqueCarrier.unique():
    PathsForMap = (data1.Origin[data.UniqueCarrier==carrier] + data1.Dest[data.UniqueCarrier==carrier])
    PathsForMap = PathsForMap.map(lambda x: sorted([x[0:3],x[3:]])).map(lambda x: x[0]+x[1])


    plt.figure(figsize=(20, 9)) 
    m = Basemap(width=7000000,height=3300000,resolution='c',projection='aea',lat_1=35.,lat_2=45,lon_0=-100,lat_0=40)
    plt.title('{} flight map'.format(carrier),fontsize=15)
    m.drawcoastlines(linewidth=0.6)
    m.drawstates(linewidth=0.2, linestyle='solid', color='k')
    m.drawcountries(linewidth=0.6)
    m.shadedrelief()


    indexes = list(PathsForMap.value_counts().index)
    values = PathsForMap.value_counts().values

    NotUS = ['SJU', 'STT', 'STX','BQN','PSE']

    for i in indexes:
        if i[:3] in NotUS or i[3:] in NotUS:
            continue
        else:
            location1LAT = airports.lat[airports.iata==i[:3]]
            location1LONG = airports.lon[airports.iata==i[:3]]
            location2LAT = airports.lat[airports.iata==i[3:]]
            location2LONG = airports.lon[airports.iata==i[3:]]

        m.drawgreatcircle(float(location1LONG),float(location1LAT),float(location2LONG),float(location2LAT),
        linewidth=1, color=(values[indexes.index(i)]/values[0],0,1-values[indexes.index(i)]/values[0]))#color=(1-((len(indexes)-indexes.index(i))/len(indexes)),0,((len(indexes)-indexes.index(i))/len(indexes)))


# As you may see on maps above, some airlines occupy whole US territory, while others focus mainly on flying on the east side of the US or flights to Alaska or other dependent territories.
# 
# I tried to modify colors on maps so the most common flights are red, and the rarer the flight is, the more blue it becomes.
# 
# 
