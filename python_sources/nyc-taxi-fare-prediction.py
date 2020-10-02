#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.plotly as py
import os
print(os.listdir("../input"))


# In[ ]:


def getDistance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon])
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))


# In[ ]:


train = pd.read_csv("../input/train.csv", nrows = 10_000)


# In[ ]:


train.isnull().sum()


# In[ ]:


train.dtypes


# In[ ]:


#transforme fields object to datetime
train.key = pd.to_datetime(train.key)
train.pickup_datetime = pd.to_datetime(train.pickup_datetime)

train['hour'] = train.pickup_datetime.dt.hour
train['day'] = train.pickup_datetime.dt.day
train['month'] = train.pickup_datetime.dt.month
train['weekday'] = train.pickup_datetime.dt.weekday

train['dist'] = getDistance(train.pickup_latitude, train.pickup_longitude, train.dropoff_latitude, train.dropoff_longitude)

train.dtypes


# In[ ]:


train.describe()


# In[ ]:


#clear dataset
train = train[train.passenger_count > 0]
train = train[(train.fare_amount > 0) & (train.fare_amount < 100)]
train = train[train.dist < 80]
train = train[(train.pickup_longitude > -80) & (train.pickup_longitude < -70)]
train = train[(train.pickup_latitude > 40) & (train.pickup_latitude < 45)]
train = train[abs(train.pickup_longitude - train.dropoff_longitude) < 5]
train = train[abs(train.pickup_latitude - train.dropoff_latitude) < 5]
train = train[train.dist > 0]


# In[ ]:


plt.hist(train.fare_amount, 100, density=True)
plt.title('Histogram - Fare Amount')
plt.grid(True)
plt.axis([0, 100, 0, 0.16])
plt.show()


# In[ ]:


plt.scatter(train.fare_amount, train.dist)
plt.show()


# In[ ]:




