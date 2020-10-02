#!/usr/bin/env python
# coding: utf-8

# #Lets begin with fetching important packages!!

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# #Have a sneak peak into the data.

# In[2]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[3]:


train.info()


# ##The data frame consists of 1458644 entities with no null values. So there is no need for fill void places.
# ###Lets check the numbers of 'Yes' entries in 'store_and_fwd_flag' and find out if we can neglect the 'N' entries so that we have a more authentic database.

# In[4]:


train[train['store_and_fwd_flag'] == 'Y'].count()


# ###It seems the case is opposite. We will take in account both types of data.
# ##Import 'gpxpy' package to calculate the distance using latitude and longitude data.
# ##Convert datetime into Hours data.

# In[5]:


import gpxpy.geo
dist = []
i = 0
for i in range(0,1458644):
    x = (gpxpy.geo.haversine_distance(train.pickup_latitude[i], train.pickup_longitude[i],
                                       train.dropoff_latitude[i], train.dropoff_longitude[i]))
    x = np.round(x/1000, decimals = 3)
    i += 1
    dist.append(x)


# In[ ]:


pick = []
for item in train['pickup_datetime']:
    item = pd.to_datetime(item).hour
    pick.append(item)
train['pick_hr'] = pd.Series(pick)

drop = []
for item in train['dropoff_datetime']:
    item = pd.to_datetime(item).hour
    drop.append(item)
train['dropoff_hr'] = pd.Series(drop)


# Lets check the dataframe along with new parameters added.

# In[7]:


train['dist'] = pd.Series(dist)
train.head(10)


# In[8]:


train['dist'].describe()


# In[9]:


fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(train['dist'], train['trip_duration'])


# ##Looks like we have entries with trip distance ranging from 0 to 1240km. This could be because of wrong coordinates entered. We can calculate number of trips which seems incorrect/rare entries e.g. trips for lesser than 50m or greater than 50km.
# ## Same is the case with time entries. We can eliminate the entries greater than 5000 sec and lesser than 30sec.

# In[10]:


train[train['dist'] < 0.05].count()


# In[11]:


train[train['dist'] > 50].count()


# In[12]:


train[train['trip_duration'] < 30].count()


# In[13]:


train[train['trip_duration'] > 5000].count()


# In[14]:


train.corr()


# #There is hardly a relation between distance and trip duration with this data (9.4%).
# ##Lets try by eliminating the entries which does not fall under a sensible trip data or which are rarely a case.. They all constitute around 1-1.5% of the total data.

# In[15]:


train1 = train[(train['dist'] > 0.05) & (train['dist'] < 26) & 
               (train['trip_duration'] > 30) & (train['trip_duration'] < 5000)]
train1.describe()


# In[16]:


fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(train1['dist'], train1['trip_duration'])


# In[17]:


train1.corr()


# #The relationship suddenly zooms to 77%. That is a significant improvement.

# In[18]:


#correlation matrix
corrmat = train1.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# ##Lets plot the distribution curves for trip time and distance.

# In[19]:


sns.distplot(train1['dist'])


# In[20]:


sns.distplot(train1['trip_duration'])


# ## Now we can state that there is a strong relationship between distance and trip duration.
# ## The other important factors to be considered here are Latitude and Longitude parameters. which defines the location of start and stop of the journey.
# #To analyse the role of coordinates, lets take subsets of the data of trips ranging a few km at a time.

# In[21]:


dummy = train1[train1['dist'] < 5]


# In[22]:


fig, ax = plt.subplots(figsize=(20,10))

ax.scatter(dummy['pickup_latitude'], dummy['trip_duration'])
ax.scatter(dummy['dropoff_latitude'], dummy['trip_duration'], marker = 'x', color = 'Red')
ax.set_xlim([40.5, 41.0])


# In[23]:


dummy1 = train1[(train1['dist'] >5) & (train1['dist'] < 10)]


# In[24]:


fig, ax = plt.subplots(figsize=(20,10))

ax.scatter(dummy1['pickup_latitude'], dummy1['trip_duration'])
ax.scatter(dummy1['dropoff_latitude'], dummy1['trip_duration'], marker = 'x', color = 'Red')
ax.set_xlim([40.5, 41.0])


# In[25]:


dummy3 = train1[(train1['dist'] >10) & (train1['dist'] < 15)]


# In[26]:


fig, ax = plt.subplots(figsize=(20,10))

ax.scatter(dummy3['pickup_latitude'], dummy3['trip_duration'])
ax.scatter(dummy3['dropoff_latitude'], dummy3['trip_duration'], marker = 'x', color = 'Red')
ax.set_xlim([40.5, 41.0])


# ##By looking at these graphs, we can say that any trip that has to pickup/drop-off or cross the area that approximately lies within latitudes 40.7-40.8 is prone to take longer journey time than usual e.g. crossing Times Square.

# In[27]:


dummy4 = train1[((train1['pickup_latitude'] < 40.7) & (train1['dropoff_latitude'] < 40.7))
                | ((train1['pickup_latitude'] > 40.8) & (train1['dropoff_latitude'] > 40.8))]


# In[28]:


fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(dummy4['dist'], dummy4['trip_duration'])


# In[29]:


dummy4.corr()


# ## This data gives a picture of short journeys that can go upto 10-12km.
# 
# To be continued.....

# In[29]:




