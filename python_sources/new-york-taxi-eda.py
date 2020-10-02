#!/usr/bin/env python
# coding: utf-8

# #Exploratory Data Analysis of the New York Taxi dataset
# ##Loading libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import time

from IPython.display import display


# In[ ]:


file=r'../input/train.csv'
train = pd.read_csv(file)
display(train.head())
#Delete ids
del train['id']
del train['store_and_fwd_flag']
del train['vendor_id']
display(train.head())
print('Number of NaN values')
display(pd.isnull(train).sum())


# ##Data cleaning

# In[ ]:


#Parsing DT string
dtparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
train['pickup_datetime'] = train['pickup_datetime'].apply(dtparse)

#Extracting DOW information 0-6 Mon-Sun
train['DOW'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].apply(lambda x:x.hour)
display(train['pickup_datetime'].describe())

#Set index later, after trimming

#Drop dropoff time
train=train.drop(['dropoff_datetime'],1)
display(train.info())


# In[ ]:


#Creating haversine distance
#<https://rosettacode.org/wiki/Haversine_formula>
from math import radians, sin, cos, sqrt, asin
def haversine(columns):
  lat1, lon1, lat2, lon2 = columns
  R = 6372.8 # Earth radius in kilometers
 
  dLat = radians(lat2 - lat1)
  dLon = radians(lon2 - lon1)
  lat1 = radians(lat1)
  lat2 = radians(lat2)
 
  a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
  c = 2*asin(sqrt(a))
 
  return R * c
cols = ['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']
distances = train[cols].apply(
    lambda x: haversine(x),axis = 1
)
train['haversine_distances'] = distances.copy()
train.head()


# In[ ]:


display(train['trip_duration'].describe())
display(train['haversine_distances'].describe())


# In[ ]:


#lowq and highq will be used throughout notebook
lowq,highq = 1,99

#Trimming based on outlier distances and durations
trip = train['trip_duration']
ltrip,rtrip= np.percentile(trip,[lowq,highq])
print(ltrip,rtrip)
train_trimmed = train[trip.between(ltrip,rtrip)]

haversine=train['haversine_distances']
ltrip,rtrip= np.percentile(haversine,[lowq,highq])
print(ltrip,rtrip)
train_trimmed = train_trimmed[haversine.between(ltrip,rtrip)]
train_trimmed.shape


# In[ ]:


display(train_trimmed.head())
display(train_trimmed.info())


# ##EDA
# ###How many values were recorded everyday?

# In[ ]:



dates=train_trimmed['pickup_datetime'].dt.date
date_sorted=dates.value_counts().sort_index()

date_sorted.plot(marker='.',ms=20,c='r',mfc='k',figsize=(16,10),
                 rot=45,lw=1,#linestyle='None'
                 title='Number of taxis driven per day',
                label='Daily')
date_rolling=date_sorted.rolling(window=7).mean()
date_rolling.plot(c='b',label='Weekly rolling average',lw=5)
plt.legend()
plt.ylabel('# of taxis')
plt.show()


# ###The dip seen at the end of January was most likely from the snowstorm, check other(better) kernels for more information
# ###Distribution of trip duration and distances

# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1)

trip_duration=train_trimmed['trip_duration']
haversine_distances=train_trimmed['haversine_distances']

sns.distplot(trip_duration,ax=ax1)
ax1.set_title('Distribution of trimmed trip duration')
ax1.set_xlabel('Trip duration(s)')

sns.distplot(haversine_distances,ax=ax2)
ax2.set_title('Distribution of trimmed haversine distances')
ax2.set_xlabel('Distance (km)')

f.tight_layout()

plt.show()


# ### What about if we log-deskew the x axis?

# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1)

trip_duration=np.log(train_trimmed['trip_duration'])
haversine_distances=np.log(train_trimmed['haversine_distances'])

sns.distplot(trip_duration,color='r',ax=ax1)
ax1.set_title('Distribution of trimmed trip duration')
ax1.set_xlabel('Trip duration(s),LOG')

sns.distplot(haversine_distances,color='r',ax=ax2)
ax2.set_title('Distribution of trimmed haversine distances')
ax2.set_xlabel('Distance (km),LOG')

f.tight_layout()

plt.show()


# ###Bivariate distribution of trip duration vs distance
# 
# As expected, the trip duration shows a positive correlation with the haversine distances. This hexbin, /w log bins, shows that the majority of trip durations are around 600s~10min and less. If anything, the hexplot distribution shows a somewhat triangular shape

# In[ ]:


sns.jointplot(y = 'trip_duration',x = 'haversine_distances',bins='log',data=train_trimmed,kind='hex',size=6,cmap='cubehelix')
plt.title('Taxicab Hexbin Trip duration vs Distance')
plt.colorbar()
plt.show()


# In[ ]:


#Log distribution of passenger count
passengers = train['passenger_count'].value_counts().sort_index()
passengers.plot(kind = 'bar',logy = True)
plt.xlabel('Number of passengers')
plt.ylabel('Frequency')
plt.title('Distribution of passenger counts, log scaling')
plt.show()


# ### How does time of day affect the average taxi duration?

# In[ ]:


grp=train_trimmed.groupby(['DOW','Hour'])['trip_duration'].mean()
taxi_tripduration=grp.unstack()

sns.heatmap(taxi_tripduration,cmap='cubehelix')
plt.xlabel('Hour picked up')
plt.ylabel('DOW picked up')
plt.title('Average trip duration (s) of taxis')
plt.show()


# ### What about the average haversine distances travelled?

# In[ ]:


grp=train_trimmed.groupby(['DOW','Hour'])['haversine_distances'].mean()
taxi_tripdistances=grp.unstack()

sns.heatmap(taxi_tripdistances,cmap='cubehelix')
plt.xlabel('Hour picked up')
plt.ylabel('DOW picked up')
plt.title('Average haversine distance (km) of taxis driven')
plt.show()


# ###### Interesting! Counter-intuitively, to me at least, the trip duration is longer despite shorter average distances. My gut-feeling explanation would be that there's alot for traffic, rush-hour between Mon-Fri 8am-6pm, so shorter distances.
# ###### Another insight is that taxis travel the longest average distances at 5am, regardless of DOW. I have no explanation for this whatsoever.
# ###What about the  number of passengers in a taxi?

# In[ ]:


grp=train_trimmed.groupby(['DOW','Hour'])['passenger_count'].mean()
taxi_tripdistances=grp.unstack()

sns.heatmap(taxi_tripdistances,cmap="YlGnBu")
plt.xlabel('Hour picked up')
plt.ylabel('DOW picked up')
plt.title('Average number of passenger in a taxi')
plt.show()


# ###On average it seems that there are more passengers picked up during weekend. My guess would be due to groups with more people partying.
# ###Trip duration vs passenger count
# It seems to show that the trip duration is longer when the driver is between pickup points

# In[ ]:


sns.lvplot(x='passenger_count', y = 'trip_duration',data = train_trimmed)
plt.xlabel('Number of passengers')
plt.ylabel('Trip duration')
plt.title('Trip duration vs passenger count')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




