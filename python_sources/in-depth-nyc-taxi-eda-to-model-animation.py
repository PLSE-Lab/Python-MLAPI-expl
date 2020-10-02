#!/usr/bin/env python
# coding: utf-8

# My very first EDA. Upvote if you find it helpful! 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import operator
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from dateutil import parser
from matplotlib import animation
import io
import base64
from IPython.display import HTML

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# ## First Look

# In[2]:


# Reading the Train Data and looking at the Given Features
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.columns)
print(train.info())
print(test.info())


# Alright so just looking at the data, we can get a sense of what this competition will entail. There are 11 columns in the train set, and 9 columns in the test set. The two columns not shared are the dropoff_datetime and trip_duration, as expected. There are no null/nan values throughout the dataset.

# In[3]:


# First Look at the Data
print('We have {} training rows and {} test rows.'.format(train.shape[0], test.shape[0]))
print('We have {} training columns and {} test columns.'.format(train.shape[1], test.shape[1]))
train.head(2)


# Based on some of the other EDAs, we already know that the IDs in both the train and test datasets are mutually exclusive, with no overlap or duplicates. As a result, we can drop the column because it won't be any help in our model throughout, and focus on the other columns in the dataset.

# ## Vendor_ID

# In[ ]:


vendor_popularity = (train['vendor_id'].value_counts())
popularity_dict = dict(vendor_popularity)

print('Most Popular Vendor:', max(vendor_popularity.iteritems(), key=operator.itemgetter(1))[0])
print('Difference in Popularity:', popularity_dict[2] - popularity_dict[1])

f = plt.figure(figsize=(10,5))
sns.barplot(vendor_popularity.index, vendor_popularity.values, alpha=0.8)
plt.xlabel('Vendor', fontsize=14)
plt.ylabel('Trips', fontsize=14)
plt.show()


# Seems pretty straightforward. There are two vendors in the dataset, with Vendor #2 having just over 100,000 more trips ordered than Vendor #1. Let's look into how their popularity changed over time. 

# In[ ]:


vendor1_change = []
vendor2_change = []

for i, row in train.iterrows():    
    
    if row['vendor_id'] == 1:
        if vendor1_change:
            list.append(vendor1_change, vendor1_change[-1] + 1)
        else:
            list.append(vendor1_change, 1)
        if vendor2_change:
            list.append(vendor2_change, vendor2_change[-1])
        else:
            list.append(vendor2_change, 0)
            
    if row['vendor_id'] == 2:
        if vendor2_change:
            list.append(vendor2_change, vendor2_change[-1] + 1)
        else:
            list.append(vendor2_change, 1)
        if vendor1_change:
            list.append(vendor1_change, vendor1_change[-1])
        else:
            list.append(vendor1_change, 0)

plt.figure(figsize=(10,5))
plt.scatter(range(train.shape[0]), vendor1_change)
plt.scatter(range(train.shape[0]), vendor2_change)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Trips Requested', fontsize=12)
plt.show()


# Nothing two interesting; both follow linear trends. I was hoping that we would see more of a exponential, logistic, or logarithmic graph, which would allow for a little more insight into the intricacies involved in the NY taxi industry throughout time.

# ## Pickup/Dropoff Datetime

# In[ ]:


# Feature Engineering
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

train['pickup_date'] = train['pickup_datetime'].dt.date
train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
train['pickup_day'] = train['pickup_datetime'].dt.day
train['pickup_month'] = train['pickup_datetime'].dt.month
train['pickup_hour'] = train['pickup_datetime'].dt.hour
train['pickup_minute'] = train['pickup_datetime'].dt.minute
train['pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).map(
    lambda x: x.total_seconds())

test['pickup_date'] = test['pickup_datetime'].dt.date
test['pickup_weekday'] = test['pickup_datetime'].dt.weekday
test['pickup_day'] = test['pickup_datetime'].dt.day
test['pickup_month'] = test['pickup_datetime'].dt.month
test['pickup_hour'] = test['pickup_datetime'].dt.hour
test['pickup_minute'] = test['pickup_datetime'].dt.minute
test['pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).map(
    lambda x: x.total_seconds())


# In[ ]:


day, count = np.unique(train['pickup_weekday'], return_counts = True)

plt.figure(figsize=(15,4))
ax = sns.barplot(x = day, y = count)
ax.set(xlabel = "Day of Week", ylabel = "Count of Taxi Rides")
plt.show();


# Seems like a pretty similar distribution, with both the weekends having fewer than average rides being requested, as expected. Not much to go off of here

# In[ ]:


day, count = np.unique(train['pickup_day'], return_counts = True)

plt.figure(figsize=(15,4))
ax = sns.barplot(x = day, y = count)
ax.set(xlabel = "Day of Month", ylabel = "Count of Taxi Rides")
plt.show();


# Also rather straightforward. Seems like a similar distribution throughout, although it drops off towards the end of the month.

# In[ ]:


day, count = np.unique(train['pickup_month'], return_counts = True)

plt.figure(figsize=(15,4))
ax = sns.barplot(x = day, y = count)
ax.set(xlabel = "Month in Year", ylabel = "Count of Taxi Rides")
plt.show();


# Not much to go off of here, 

# In[ ]:


day, count = np.unique(train['pickup_hour'], return_counts = True)

plt.figure(figsize=(15,4))
ax = sns.barplot(x = day, y = count)
ax.set(xlabel = "Hour in Day", ylabel = "Count of Taxi Rides")
plt.show();


# Again, nothing strange. Much fewer rides are requested in the early hours of the day, with normal hours being from 7 A.M to 12 A.M.

# ## Passenger Count

# In[ ]:


passengers, count = np.unique(train['passenger_count'], return_counts = True)
passenger_count = train['passenger_count'].value_counts()
print(passenger_count)

plt.figure(figsize=(15,4))
ax = sns.barplot(x = passengers, y = count)
ax.set(xlabel = "Number of Passengers", ylabel = "Count of Taxi Rides")
plt.show();


# The chart goes to show that a ride is most often requested by a single passenger, which is expected. However, there were also 60 rides requested in which there were zero passengers, and 5 others in which there were seven, eight, or nine passengers. We'll plan what to do with these outliers later.

# ## Pickup/Dropoff Longitude/Latitude

# In[ ]:


# Pickup Latitude/Longitude
sns.lmplot(x="pickup_longitude", y="pickup_latitude", fit_reg=False, 
           size=9, scatter_kws={'alpha':0.3,'s':5}, data=train[(
                 train.pickup_longitude>train.pickup_longitude.quantile(0.005))
               &(train.pickup_longitude<train.pickup_longitude.quantile(0.995))
               &(train.pickup_latitude>train.pickup_latitude.quantile(0.005))                           
               &(train.pickup_latitude<train.pickup_latitude.quantile(0.995))])

plt.xlabel('Pickup Longitude');
plt.ylabel('Pickup Latitude');
plt.show()


# This shows the overall distribution of pickup points across New York. I dropped a few obvious outliers/errors to make the graphic easier to make out.

# In[ ]:


# Dropoff Latitude/Longitude
sns.lmplot(x="dropoff_longitude", y="dropoff_latitude", fit_reg=False, 
           size=9, scatter_kws={'alpha':0.3,'s':5}, data=train[(
                 train.dropoff_longitude>train.dropoff_longitude.quantile(0.005))
               &(train.dropoff_longitude<train.dropoff_longitude.quantile(0.995))
               &(train.dropoff_latitude>train.dropoff_latitude.quantile(0.005))                           
               &(train.dropoff_latitude<train.dropoff_latitude.quantile(0.995))])

plt.xlabel('Dropoff Longitude');
plt.ylabel('Dropoff Latitude');
plt.show()


# This shows the overall distribution of dropoff points across New York. I also dropped a few obvious outliers/errors to make the graphic easier to make out. 
# 
# Next I'll explore pickup points when compared with the date.

# In[ ]:


fig = plt.figure(figsize = (10,10))
ax = plt.axes()


# In[ ]:


# Weekday Pickup Latitude/Longitude
# Credit to DrGuillermo for the Animation idea

def pickup_weekday(day):
    ax.clear()
    ax.set_title('Pickup Locations - Day ' + str(int(day)))    
    plt.figure(figsize = (8,10))
    temp = train[train['pickup_weekday'] == day]
    temp = temp[(
        train.pickup_longitude>train.pickup_longitude.quantile(0.005))
      &(train.pickup_longitude<train.pickup_longitude.quantile(0.995))
      &(train.pickup_latitude>train.pickup_latitude.quantile(0.005))                           
      &(train.pickup_latitude<train.pickup_latitude.quantile(0.995))]
    ax.plot(temp['pickup_longitude'], temp['pickup_latitude'],'.', 
            alpha = 1, markersize = 2, color = 'gray')

ani = animation.FuncAnimation(fig,pickup_weekday,sorted(train.pickup_weekday.unique()), interval = 1000)
ani.save('animation.gif', writer='imagemagick', fps=2)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# This shows an animation of all the pickup locations, separated by their respective day of the week. I was hoping that it would show a bit more, although it seems to be rather uniform throughout each day of the week, with no obvious trends throughout the visualization

# In[ ]:


# Weekday Pickup Latitude/Longitude
# Credit to DrGuillermo for the Animation idea

def pickup_hour(hour):
    ax.clear()
    ax.set_title('Pickup Locations - Hour ' + str(int(hour)))    
    plt.figure(figsize = (8,10))
    temp = train[train['pickup_hour'] == hour]
    temp = temp[(
        train.pickup_longitude>train.pickup_longitude.quantile(0.005))
      &(train.pickup_longitude<train.pickup_longitude.quantile(0.995))
      &(train.pickup_latitude>train.pickup_latitude.quantile(0.005))                           
      &(train.pickup_latitude<train.pickup_latitude.quantile(0.995))]
    ax.plot(temp['pickup_longitude'], temp['pickup_latitude'],'.', 
            alpha = 1, markersize = 2, color = 'gray')

ani = animation.FuncAnimation(fig,pickup_hour,sorted(train.pickup_hour.unique()), interval = 1000)
ani.save('animation.gif', writer='imagemagick', fps=2)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# Definitely more interesting here, especially when looking at the density of the responses as the day goes on. In some cases, remote locations get more densely populated with responses before others, especially in the early hours of the day, which I infer could correspond to places where people live. As the day goes on, people travel into the city to work, and request rides back as the day comes to an end. If I were to create an animation of the drop-off location in regards to the hour of the day, I expect the graph to be almost an inverse of this one.

# In[ ]:


# Feature Engineering (Credit to Beluga)
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train['distance_haversine'] = haversine_array(
    train['pickup_latitude'].values, train['pickup_longitude'].values,
    train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train['distance_dummy_manhattan'] = dummy_manhattan_distance(
    train['pickup_latitude'].values, train['pickup_longitude'].values,
    train['dropoff_latitude'].values, train['dropoff_longitude'].values)

test['distance_haversine'] = haversine_array(
    test['pickup_latitude'].values, test['pickup_longitude'].values,
    test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test['distance_dummy_manhattan'] = dummy_manhattan_distance(
    test['pickup_latitude'].values, test['pickup_longitude'].values,
    test['dropoff_latitude'].values, test['dropoff_longitude'].values)

train['avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
train['avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']

train['center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
train['center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
test['center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
test['center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2

train['pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)
train['pickup_long_bin'] = np.round(train['pickup_longitude'], 2)
train['center_lat_bin'] = np.round(train['center_latitude'], 2)
train['center_long_bin'] = np.round(train['center_longitude'], 2)
train['pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600))
test['pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)
test['pickup_long_bin'] = np.round(test['pickup_longitude'], 2)
test['center_lat_bin'] = np.round(test['center_latitude'], 2)
test['center_long_bin'] = np.round(test['center_longitude'], 2)
test['pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600))

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, 
                                          train['pickup_longitude'].values, 
                                          train['dropoff_latitude'].values, 
                                          train['dropoff_longitude'].values)

test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, 
                                         test['pickup_longitude'].values, 
                                         test['dropoff_latitude'].values, 
                                         test['dropoff_longitude'].values)


# ## PCA

# In[ ]:


# Feature Engineering (Credit to Beluga)
full = pd.concat([train, test])
coords = np.vstack((full[['pickup_latitude', 'pickup_longitude']], 
                   full[['dropoff_latitude', 'dropoff_longitude']]))

pca = PCA().fit(coords)
train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

train['pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) +                          np.abs(train['dropoff_pca0'] - train['pickup_pca0'])

test['pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) +                         np.abs(test['dropoff_pca0'] - test['pickup_pca0'])


# ## Correlation

# In[ ]:


corr = train.corr()
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()


# Unsurprisingly, not many of the features are correlated with one another, being that they represent different things. A few of our derived distance features and pca_manhattan have a little correlation with trip_duration, which will probably prove to be helpful as we continue to explore the data.

# ## Model

# In[ ]:





# ## Continuation

# I'll be coming back to this EDA later, adding more and more as the competion goes on. For now though, to see how the derived features do on the leaderboard, check out my take on a stacking model: https://www.kaggle.com/misfyre/stacking-model-378-lb-375-cv
# 
# Thanks for reading! If this helped you, please upvote.
