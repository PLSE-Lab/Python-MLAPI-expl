#!/usr/bin/env python
# coding: utf-8

# I am going to play with the ride time in this kernel and see how that associates with the duration, or more specifically, the speed of the ride.
# As this data set came from NYC, I am going to calculate the distance by Manhattan distance for an obvious reason -
#  the distance of the trip can be better approached by Manhattan distance rather than Euclidean distance.
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
import matplotlib.pyplot as plt
import datetime


# In[2]:


train = pd.read_csv("../input/train.csv")


# Alright, to start with this there are some basic stuff I would like to go over and see if we want to do some engineering or cleaning.

# In[3]:


display(train.head())
print(train.shape)
print(train.isnull().sum())


# In[ ]:





# In[4]:


fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(train.pickup_longitude, train.pickup_latitude, s=0.05, cmap='jet')
ax.scatter(train.dropoff_longitude, train.dropoff_latitude, s=0.05, cmap='jet')

plt.show();


# Awesome, there is no null value in the data set.  
# There are some potential cleaning can be done here, but we will move on to the parts that we are focusing here without further due. 
# Duration should not be compared directly since travel distance definitely plays a huge role here. That being said, I am adding a new column that divides duration by total traveling distance. The larger this value is the slower the traffic is.
# I am going to add longitude difference and latitude difference just in case we want to dig more into it in the next phase.

# In[11]:


# Datetime features
train["pickup_datetime"]=pd.to_datetime(train['pickup_datetime'])
train["dropoff_datetime"]=pd.to_datetime(train['dropoff_datetime'])

# Distance feature
train["longitude_diff"] = train.dropoff_longitude - train.pickup_longitude
train["latitude_diff"] = train.dropoff_latitude -train.pickup_latitude


# In[12]:


display(train.shape)
display(train.head())


# Oops, the difference between longitude and latitude might be too small. The difference of speed might not be that obvious. Let's scale it a bit

# In[13]:


#duration/distance
train['duration_dist'] = train.trip_duration/100*(abs(train.longitude_diff) + abs(train.latitude_diff))
train['duration_dist'] = (train.duration_dist - train.duration_dist.mean())/train.duration_dist.std()


# In[14]:


display(train.duration_dist.describe())


# In[15]:


train['pickup_t'] = (train.pickup_datetime.dt.hour * 3600 + train.pickup_datetime.dt.minute * 60 + train.pickup_datetime.dt.second)/3600
train['dropoff_t'] = (train.dropoff_datetime.dt.hour * 3600 + train.dropoff_datetime.dt.minute * 60 + train.dropoff_datetime.dt.second)/3600


# In[16]:


train.plot.scatter('pickup_t', 'dropoff_t', figsize=(7,7),c='duration_dist', cmap='jet');


# So it looks like pickup time and dropoff time doesn't affect the traffic that much. I assume the traffic will be slower in the city, but it might not be obvious if we only plot pickup and dropoff location...

# In[19]:


# Kudos to kaggler DrGuillermo
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]
train1 = train[(train.pickup_longitude> xlim[0]) & (train.pickup_longitude < xlim[1])]
train1 = train1[(train1.dropoff_longitude> xlim[0]) & (train1.dropoff_longitude < xlim[1])]
train1 = train1[(train1.pickup_latitude> ylim[0]) & (train1.pickup_latitude < ylim[1])]
train1 = train1[(train1.dropoff_latitude> ylim[0]) & (train1.dropoff_latitude < ylim[1])]
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(train1.pickup_longitude, train1.pickup_latitude, s=0.001,c=train1.duration_dist, cmap='jet')
ax.scatter(train1.dropoff_longitude, train1.dropoff_latitude, s=0.001,c=train1.duration_dist, cmap='jet')


plt.show();


# In[20]:


train_all = train[ abs(train.duration_dist - train.duration_dist.mean() ) < ( 3  * train.duration_dist.std() )]


# In[21]:


train_all.plot.scatter('pickup_t','dropoff_t',s=1, figsize = (7,7), c='duration_dist',colormap='jet');


# Ok, there are some slower rides during the day. I guess in NYC rush hour starts from 7 am and ends around 8 pm.
# What else caught my attention is the data points scattered at the bottom of the graph. They are rides that are dropped off around midnight but the pickup time scatters across a day. Let's see how it'll be when we put week of day in the plot

# In[22]:


p_time = (train_all.pickup_datetime.dt.weekday * 24 + (train_all.pickup_datetime.dt.hour * 3600 + train_all.pickup_datetime.dt.minute * 60 + train_all.pickup_datetime.dt.second)/3600)/24
d_time = (train_all.dropoff_datetime.dt.weekday * 24 + (train_all.dropoff_datetime.dt.hour * 3600 + train_all.dropoff_datetime.dt.minute * 60 + train_all.dropoff_datetime.dt.second)/3600)/24
c = train_all.duration_dist
fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(p_time, d_time,s=3, c=c, cmap='jet')

plt.show();


# So there are some patterns here. A lot of short trips that are pickup and drop off immediately or even simultaneously. There is a line of slower rides that are picked up then dropped off the next day around the same time. Now it's getting suspicious. There is also a pattern that some trips that start during a day will end around the following midnight, which is not that obvious.
# 
# There might be some other factor caused this.
# I saw some discussion that the direction might affect the traveling speed for the East-West direction streets are usually shorter and traveling along streets will encounter more traffic lights. 
# I also want to see if store and forward has some effect here.

# In[23]:


train_all=pd.get_dummies(train_all, columns=["store_and_fwd_flag"])


# In[25]:


plt.clf()
c = train_all.duration_dist
d = train_all.store_and_fwd_flag_Y
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(d, c)

plt.show();


# There are difference but not really obvious...

# In[26]:


plt.clf();
x = (train_all.longitude_diff - train_all.longitude_diff.mean())/train_all.longitude_diff.std()
y = (train_all.latitude_diff - train_all.latitude_diff.mean())/train_all.latitude_diff.std()
c = train_all.duration_dist
fig, ax = plt.subplots(figsize = (7,7))
ax.scatter(x,y,s=1, c=c, cmap='jet')
plt.show();


# It is not very clear whether it has any effect on the traveling speed.
