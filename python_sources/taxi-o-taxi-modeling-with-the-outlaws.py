#!/usr/bin/env python
# coding: utf-8

# Here I will be doing some outlier detection.

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the "../input/" directory.
path = 'D:/BACKUP/Kaggle/New York City Taxi/Data/'
train_df = pd.read_csv('../input/train.csv')

#--- Let's peek into the data
print (train_df.head())


# In[ ]:


#--- The test data ---
test_df = pd.read_csv('../input/test.csv')

#--- Check if there are any Nan values ---
print (test_df.isnull().values.any())

#--- Let's peek into the data
print (test_df.head())


# In[ ]:


#--- Add columns for months, days and hours for pickup and drop off ---
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

train_df['pickup_month'] = train_df.pickup_datetime.dt.month.astype(np.uint8)
train_df['pickup_day'] = train_df.pickup_datetime.dt.weekday.astype(np.uint8)
train_df['pickup_hour'] = train_df.pickup_datetime.dt.hour.astype(np.uint8)

train_df['dropoff_month'] = train_df.dropoff_datetime.dt.month.astype(np.uint8)
train_df['dropoff_day'] = train_df.dropoff_datetime.dt.weekday.astype(np.uint8)
train_df['dropoff_hour'] = train_df.dropoff_datetime.dt.hour.astype(np.uint8)
print (train_df.head())


# In[ ]:


#--- Add columns for months, days and hours for pickup ONLY for test set---
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])

test_df['pickup_month'] = test_df.pickup_datetime.dt.month.astype(np.uint8)
test_df['pickup_day'] = test_df.pickup_datetime.dt.weekday.astype(np.uint8)
test_df['pickup_hour'] = test_df.pickup_datetime.dt.hour.astype(np.uint8)

print (test_df.head())


# In[ ]:


#--- Let us also add the displacement column ---

from math import radians, cos, sin, asin, sqrt   #--- for the mathematical operations involved in the function ---

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

train_df['Displacement (km)'] = train_df.apply(lambda x: haversine(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)
test_df['Displacement (km)'] = test_df.apply(lambda x: haversine(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)

print (train_df.head())
print (test_df.head())


# Add another column for trip duration in minutes
# 

# In[ ]:


train_df['trip_duration_mins'] = train_df.apply(lambda row: row['trip_duration'] / 60, axis=1)

print (train_df.head())

print (len(train_df))


# From our analysis in Part 1, we will find the range (minimum and maximum) trip duration

# In[ ]:


print (max(train_df['trip_duration_mins'].values))
print (min(train_df['trip_duration_mins'].values))


# In[ ]:


#--- Count number of trip durations above 60 mintues ---
print (train_df.id[(train_df['trip_duration_mins'] > 60)].count())


# In[ ]:


print (train_df.id.count())


# Out of the 1458644 observations, 12317 are more than 1 hour

# In[ ]:


#--- Count number of trip durations above 2 hours ---
print (train_df.id[(train_df['trip_duration_mins'] > 120)].count())


# In[ ]:


#--- Count number of trip durations above 3 hours ---
print (train_df.id[(train_df['trip_duration_mins'] > 180)].count())


# In[ ]:


#--- Count number of trip durations above 4 hours ---
print (train_df.id[(train_df['trip_duration_mins'] > 240)].count())


# In[ ]:


#--- Count number of trip durations above 5 hours ---
print (train_df.id[(train_df['trip_duration_mins'] > 300)].count())


# In[ ]:


print(2072/1458644)


# Let us visualize the pick and dropoff locations on a scatter plot

# In[ ]:


plt.plot(train_df['pickup_longitude'], train_df['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Pickup Location Lat and Long', weight = 'bold')
plt.show()

plt.plot(train_df['dropoff_longitude'], train_df['dropoff_latitude'], '.', color='k', alpha=0.8)
plt.title('Dropoff Location Lat and Long', weight = 'bold')
plt.show()


# Let us remove occurences more than 5 hours of trip duration and visualize the plot again

# In[ ]:


reduced_df = train_df[train_df.trip_duration_mins < 300]
print(len(reduced_df))


# By removing occurences above 5 hours of trip duration we still hold 99.857% of the actual occurences.
# 
# Now let us visualize the plot again

# In[ ]:


plt.plot(reduced_df['pickup_longitude'], reduced_df['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Pickup Location Lat and Long', weight = 'bold')
plt.show()

plt.plot(reduced_df['dropoff_longitude'], reduced_df['dropoff_latitude'], '.', color='k', alpha=0.8)
plt.title('Dropoff Location Lat and Long', weight = 'bold')
plt.show()


# The plot still appears to be the same. 
# 
# So we can conclude that, duration of the trip has **NO** relation with pickup and dropoff locations.

# From both the plots we can clearly see that there are some extreme points, which has to be knocked off to fit a better model.

# In[ ]:


train_df = train_df[train_df.pickup_latitude != 51.881084442138672]

train_df = train_df[train_df.pickup_longitude != -121.93334197998048]

train_df = train_df[train_df.dropoff_longitude != -121.93320465087892]

train_df = train_df[train_df.dropoff_latitude != 32.181140899658203]


# In[ ]:


plt.plot(train_df['pickup_longitude'], train_df['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Pickup Location Lat and Long', weight = 'bold')
plt.show()

plt.plot(train_df['dropoff_longitude'], train_df['dropoff_latitude'], '.', color='k', alpha=0.8)
plt.title('Dropoff Location Lat and Long', weight = 'bold')
plt.show()


# Maybe if we drop rows having extreme trip durations we can reduce this plot above.

# Vertically stacking both pickup and dropoff locations respectively under latitude and longitude
# 
# We are going to normalize the latitude and logitude positions for pickup and dropoff locations collectively.

# In[ ]:


pickup = train_df[['pickup_latitude', 'pickup_longitude']]
dropoff = train_df[['dropoff_latitude', 'dropoff_longitude']]

dropoff.columns = ['pickup_latitude', 'pickup_longitude']
frames = [pickup, dropoff]

locations = pd.concat(frames)

print (pickup.shape)
print (dropoff.shape)

print (locations.shape)


# In[ ]:


#--- Mean of locations Lats and Longs ---
mean_loc_lat = np.mean(locations['pickup_latitude'])
mean_loc_lon = np.mean(locations['pickup_longitude'])

print (mean_loc_lat)
print (mean_loc_lon)


# In[ ]:


#--- Standard deviation of pickup & dropoff Lats and Longs ---
std_locations_lat = np.std(locations['pickup_latitude'])
std_locations_lon = np.std(locations['pickup_longitude'])

print (std_locations_lat)
print (std_locations_lon)


# In[ ]:


#--- Find the range to plot 
#--- using Mean +/- std

min_loc_lat = mean_loc_lat - (3 * std_locations_lat)
max_loc_lat = mean_loc_lat + (3 * std_locations_lat)
min_loc_lon = mean_loc_lon - (3 * std_locations_lon)
max_loc_lon = mean_loc_lon + (3 * std_locations_lon)


# In[ ]:



locations = locations[(locations.pickup_latitude > min_loc_lat) & (locations.pickup_latitude < max_loc_lat) & (locations.pickup_longitude > min_loc_lon) & (locations.pickup_longitude < max_loc_lon)]

print (locations.shape)


# Let us visualize the pickup and dropoff locations again after normalization

# In[ ]:


plt.plot(locations['pickup_longitude'], locations['pickup_latitude'], '.', color='k', alpha=0.8)
plt.title('Location Lat and Long', weight = 'bold')
plt.show()


# The plot above is for both pickup and dropoff locations combined.
# 
# We have to reduce **train_df** based on the values present in **locations** df. 

# #Its REGRESSION TIME !!!!

# I want to start off with a simple regression without removing any observation.
# 
# Here the features I want to predict the trip duration against.

# In[ ]:


#--- Convert store 'store_and_flag_fwd' to a numerical variable --

train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].astype('category')
train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].cat.codes

test_df['store_and_fwd_flag'] = test_df['store_and_fwd_flag'].astype('category')
test_df['store_and_fwd_flag'] = test_df['store_and_fwd_flag'].cat.codes


# In[ ]:


features = train_df[['vendor_id','Displacement (km)', 'pickup_hour','pickup_month','pickup_day','passenger_count','dropoff_latitude','dropoff_longitude','pickup_latitude','pickup_longitude','store_and_fwd_flag']]
target = train_df[['trip_duration']]


# ##Model 1

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

reg = linear_model.LinearRegression()
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(reg, features, target, cv=cv)


# In[ ]:


reg = linear_model.Ridge (alpha = .5)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(reg, features, target, cv=cv)


# In[ ]:


reg = linear_model.Ridge (alpha = .01)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(reg, features, target, cv=cv)   


# In[ ]:


reg.fit(features,target)


# In[ ]:


#--- Choose the same features as the ones chosen from the train_df for training ---
tfeatures = test_df[['vendor_id','Displacement (km)', 'pickup_hour', 'pickup_month','pickup_day','passenger_count','dropoff_latitude','dropoff_longitude','pickup_latitude','pickup_longitude','store_and_fwd_flag']]


# In[ ]:


#--- Predict the test data ---
pred = reg.predict(tfeatures)


# In[ ]:


#--- append the predicted trip duration to test_df ---
test_df['trip_duration']=pred.astype(int)
out = test_df[['id','trip_duration']]


# In[ ]:


#--- Check for any Nan /missing values in the predicted output ---
out['trip_duration'].isnull().values.any()


# In[ ]:


out.to_csv('linear_Ridge_1.csv',index=False) 


# ##Model 2

# In[ ]:


reg = linear_model.Lasso (alpha = .01)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
cross_val_score(reg, features, target, cv=cv)


# In[ ]:


reg.fit(features,target)
pred = reg.predict(tfeatures)

test_df['trip_duration']=pred.astype(int)
out = test_df[['id','trip_duration']]

out['trip_duration'].isnull().values.any()

out.to_csv('linear_Lasso_2.csv',index=False) 

 


# ##Model 3

# In[ ]:


features = train_df[['Displacement (km)', 'pickup_hour','pickup_month','pickup_day','dropoff_latitude','dropoff_longitude','pickup_latitude','pickup_longitude']]
target = train_df[['trip_duration']]

reg = linear_model.Ridge (alpha = .5)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
cross_val_score(reg, features, target, cv=cv)


# Let's use a reduced set of features: removed '**vendor_id**', '**passenger_count**' and 'store_and_fwd_flag'

# In[ ]:


#--- Choose the same features as the ones chosen from the train_df for training ---
tfeatures = test_df[['Displacement (km)', 'pickup_hour', 'pickup_month','pickup_day','dropoff_latitude','dropoff_longitude','pickup_latitude','pickup_longitude']]

reg.fit(features,target)
pred = reg.predict(tfeatures)

test_df['trip_duration']=pred.astype(int)
out = test_df[['id','trip_duration']]

out['trip_duration'].isnull().values.any()

out.to_csv('linear_Lasso_3_reduced.csv',index=False) 


# ##Choosing good correlated features without reduction

# In[ ]:


train_df[train_df.columns[1:]].corr()['trip_duration'][:-1]
#train_df.head()


# ##Model 4
# 
# Let us model one having only good correlated variables

# In[ ]:


features = train_df[['vendor_id','Displacement (km)', 'dropoff_latitude','dropoff_longitude','pickup_latitude','pickup_longitude']]
target = train_df[['trip_duration']]

reg = linear_model.Lasso (alpha = .5)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
cross_val_score(reg, features, target, cv=cv)


# In[ ]:


#--- Choose the same features as the ones chosen from the train_df for training ---
tfeatures = test_df[['vendor_id','Displacement (km)','dropoff_latitude','dropoff_longitude','pickup_latitude','pickup_longitude']]

reg.fit(features,target)
pred = reg.predict(tfeatures)

test_df['trip_duration']=pred.astype(int)
out = test_df[['id','trip_duration']]

out['trip_duration'].isnull().values.any()

out.to_csv('linear_Lasso_4_corr_reduced.csv',index=False) 


# Adding the two new features from previous kernel:
# 
#  1. Manhattan Distance
#  2. Bearing Distance.

# In[ ]:


def arrays_bearing(lats1, lngs1, lats2, lngs2, R=6371):
    lats1_rads = np.radians(lats1)
    lats2_rads = np.radians(lats2)
    lngs1_rads = np.radians(lngs1)
    lngs2_rads = np.radians(lngs2)
    lngs_delta_rads = np.radians(lngs2 - lngs1)
    
    y = np.sin(lngs_delta_rads) * np.cos(lats2_rads)
    x = np.cos(lats1_rads) * np.sin(lats2_rads) - np.sin(lats1_rads) * np.cos(lats2_rads) * np.cos(lngs_delta_rads)
    
    return np.degrees(np.arctan2(y, x))

train_df['bearing_dist'] = arrays_bearing(
train_df['pickup_latitude'], train_df['pickup_longitude'], 
train_df['dropoff_latitude'], train_df['dropoff_longitude'])

print (train_df.head())


# In[ ]:


train_df['Manhattan_dist'] =     (train_df['dropoff_longitude'] - train_df['pickup_longitude']).abs() +     (train_df['dropoff_latitude'] - train_df['pickup_latitude']).abs()
    
print(train_df.head()) 


# #Modeling by REMOVING the outliers !!!

# In[ ]:





# ##There are still many more models to come
# 
# #STAY TUNED
