#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import *

# this is to turn on the option to display all columns when we get a sample of data using df.head()
set_option('display.max_columns', None)
set_option('display.max_rows', None) 

df = read_csv("../input/train.csv")
print("Finished loading dataset!")


# Now let's get a sense of what the dataset looks at using df.head(). You can input a parameter into this function to indicate how many rows you would like to see.

# In[ ]:


df.head()


# We can use df.shape to know the shape of the dataset (how many rows and how many columns). Each row is a data entry and each column is a feature of the dataset.

# In[ ]:


df.shape


# In[ ]:


# for purpose of demonstration, we will only take the first 10000 rows 
df = df.head(10000)


# In[ ]:


# Ex: to access the dropoff_latitude of the first entry(row)
df.iloc[0]['dropoff_latitude']


# In[ ]:


from math import sin, cos, sqrt, atan2, radians

def getDistance(lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance
# add a new column called trip_distance, the unit is km
df['trip_distance'] = 0.0
df.head()


# In[ ]:


for (index, row) in df.iterrows():
    pickup_lon = row['pickup_longitude']
    pickup_lat = row['pickup_latitude']
    dropoff_lon = row['dropoff_longitude']
    dropoff_lat = row['dropoff_latitude']
    df.at[index, 'trip_distance'] = getDistance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
import matplotlib.pyplot as plt
df.head(10)


# In[ ]:


import matplotlib.pyplot as plt
trip_duration_list = []
trip_distance_list = []
for (index, row) in df.iterrows():
    trip_duration_list.append(row['trip_duration'])
    trip_distance_list.append(row['trip_distance'])
plt.plot(trip_distance_list, trip_duration_list, 'ro', markerSize = 1)
plt.show()


# We can see that there are a couple outliers from this graph. The clustering at the upper left corner likely corresponds to trips that stayed at a place for a long time without moving. We consider removing these outliers and plot the data again.

# In[ ]:


trip_duration_list = []
trip_distance_list = []
for (index, row) in df.iterrows():
    if row['trip_duration'] > 20000: continue
    trip_duration_list.append(row['trip_duration'])
    trip_distance_list.append(row['trip_distance'])
plt.plot(trip_distance_list, trip_duration_list, 'ro', markerSize = 1)
plt.show()


# Now we can see a better graph that displays the relationship between trip duration and trip distance. We can clearly see a positive correlation from this graph - the longer the distance of the trip, the longer the trip takes. This totally makes sense. We will try to fit a line on this graph.

# In[ ]:


import numpy as np

from sklearn import linear_model

num_trips = len(trip_distance_list)

Y = np.array(trip_duration_list)
X = np.ndarray(shape=(num_trips, 1), dtype=float)
for i in range(num_trips):
    X[i][0] = trip_distance_list[i]
    
clf = linear_model.LinearRegression()
clf.fit(X, Y)
print("Slope: " + str(clf.coef_[0]))
print("Intercept: " + str(clf.intercept_))


# In[ ]:




