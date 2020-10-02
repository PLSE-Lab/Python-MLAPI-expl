#!/usr/bin/env python
# coding: utf-8

# #### First, import some basic modules

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Load the training data. 
# There're 1458644 instances. 
# The datetime format also needs to be converted.

# In[ ]:


train = pd.read_csv("../input/train.csv")
train.tail()


# #### Import the datetime module and do the transformation

# In[ ]:


import datetime
pickup = [datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in train['pickup_datetime']]
pickup[1]


# #### Check the year of the instances.

# In[ ]:


pickup_year = pd.Series(i.year for i in pickup)
pickup_year.value_counts()


# #### All the data collected are rides in 2016, we ignore the information about year.

# #### Check the month of the instances.
# The rides are between Jan and June.

# In[ ]:


pickup_month = pd.Series(i.month for i in pickup)
pickup_month.value_counts()


# #### Append the month,day,hour,minute and second information extracted to the train dataframe.

# In[ ]:


train['pickup_month'] = pd.Series(i.month for i in pickup)
train['pickup_day'] = pd.Series(i.day for i in pickup)
train['pickup_hour'] = pd.Series(i.hour for i in pickup)
train['pickup_minute'] = pd.Series(i.minute for i in pickup)
train['pickup_second'] = pd.Series(i.second for i in pickup)
train.head()


# #### We can do the similar datetime conversion for the pickup_datetime column.

# In[ ]:


dropoff = [datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in train['dropoff_datetime']]
train['dropoff_day'] = pd.Series(i.day for i in dropoff)
train['dropoff_hour'] = pd.Series(i.hour for i in dropoff)
train['dropoff_minute'] = pd.Series(i.minute for i in dropoff)
train['dropoff_second'] = pd.Series(i.second for i in dropoff)
train.head()


# #### The characters in the "store_and_fwd_flag" column also need to converted to integers.

# In[ ]:


train['store_and_fwd_flag_int'] = pd.Series(int(i=='Y') for i in train['store_and_fwd_flag'])


# Then the original "pickup_datetime", "dropoff_datetime" and "store_and_fwd_flag" columns can be dropped.

# In[ ]:


del train['pickup_datetime']
del train['dropoff_datetime']
del train['store_and_fwd_flag']
train.head()


# #### Let's plot the scattering plot between "passenger_count" and "trip_duration" to see if there's any outliers.

# In[ ]:


plt.scatter(train['passenger_count'],train['trip_duration'])


# #### Looks like there're 4 outliers. We can pickout the 4 instances and examine them. Probably some error in the pickup_day & dropoff_day data. 

# In[ ]:


abnormal = train[train['trip_duration'] > 1500000]
abnormal


# #### We can go ahead drop those outliers and instances in which the passenger_count = 0

# In[ ]:


train = train[train['trip_duration'] < 1500000]
train = train[train['passenger_count'] > 0]


# #### Let's view the possible relationship between passenger_count and trip_duration.

# In[ ]:


plt.scatter(train['passenger_count'],train['trip_duration'])


# #### Something fishy about instances that have passenger_count > 6. We can pick them out to take a look.

# In[ ]:


abnormal2 = train[train['passenger_count'] > 6]
abnormal2


# In[ ]:


plt.scatter(train['pickup_longitude'],train['trip_duration'])


# #### Most of the data are about rides with longitudes between -80 and -60. Two of the rides are at somewhere else.

# In[ ]:


abnormal3 = train[train['pickup_longitude'] < -100]
abnormal3


# #### View the relationship between other time factors and trip_duration.

# In[ ]:


fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(231)
plt.scatter(train['pickup_month'],train['trip_duration'])
ax2 = fig.add_subplot(232)
plt.scatter(train['pickup_day'],train['trip_duration'])
ax3 = fig.add_subplot(233)
plt.scatter(train['pickup_hour'],train['trip_duration'])
ax4 = fig.add_subplot(234)
plt.scatter(train['pickup_minute'],train['trip_duration'])
ax5 = fig.add_subplot(235)
plt.scatter(train['pickup_second'],train['trip_duration'])

