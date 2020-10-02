#!/usr/bin/env python
# coding: utf-8

# ***Introduction***

# In *New York City Taxi Trip Duration*, we have to predict the trip_duration in NYC. Trip_duration is depends upon from where the taxi picks the passengers and where to drop. We have DataTime stamp in the training as well as testing data. We have pickup and dropoff locations, pasenger count. 
# Evaluation creteria is RMSLE. We have trained our model using xgboost.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import timedelta 
import datetime as dt
import math
from haversine import haversine # Distance Calculation
import matplotlib.pyplot as plt # plots
plt.rcParams['figure.figsize'] = [16, 10]
import seaborn as sns
import xgboost as xgb  # regressor
from sklearn.model_selection import train_test_split # for spliting data
from sklearn.linear_model import LinearRegression 
from sklearn.decomposition import PCA 
from sklearn.cluster import MiniBatchKMeans
import pickle
import warnings
warnings.filterwarnings('ignore')


# Input files. N= 100000 for the EDA

# In[2]:


np.random.seed(1987)
N = 100000 # number of sample rows in plots
t0 = dt.datetime.now()
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')


# In[3]:


print('training rows: {} and test rows:{} '.format(train.shape[0], test.shape[0]))
print('training columns: {} and  test columns:{}'.format(train.shape[1], test.shape[1]))
print('in training file dropoff_time will be droped while training the model')
train.head(10)


# In[4]:


print('Training and testing files are mutualy exclusive. Id is unique so it wont be contributing any part in model training. Only 2 vendors are there in data set.')


# In[5]:


train['trip_duration'].max()/3600
print('max trip_duration in hours are 979.52')


# In[6]:


df = train
# Convert the date to a pandas datetime format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format="%Y/%m/%d %H:%M:%S")
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], format="%Y/%m/%d %H:%M:%S")

# Pull out the month, day of week and hour of day and make a new feature for each
df['month'] = df['pickup_datetime'].dt.month
df['dow'] = df['pickup_datetime'].dt.dayofweek
df['hour'] = df['pickup_datetime'].dt.hour

# Count number of pickups made per month, day of week and hour of day
month_usage = pd.value_counts(df['month']).sort_index()
dow_usage = pd.value_counts(df['dow']).sort_index()
hour_usage = pd.value_counts(df['hour']).sort_index()


# In[7]:


# Set custom xtick labels
x_tick_labels_month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
x_tick_labels_day = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# define subplot
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(19, 15))

figure = plt.subplot(2, 2, 1)
month_usage.plot.bar(alpha = 0.5, color = 'green')
plt.title('Pickups over Month of Year', fontsize = 20)
plt.xlabel('Month', fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.xticks(month_usage.index - 1, x_tick_labels_month, rotation='90', fontsize=18)
plt.xticks(rotation=0)
plt.yticks(fontsize = 18)

figure = plt.subplot(2, 2, 2)
dow_usage.plot.bar(alpha = 0.5, color = 'blue')
plt.title('Pickups over Day of Week', fontsize = 20)
plt.xlabel('Day of Week', fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.xticks(dow_usage.index, x_tick_labels_day, rotation='90', fontsize=18)
plt.xticks(rotation=0)
plt.yticks(fontsize = 18)

figure = plt.subplot(2, 1, 2)
hour_usage.plot.bar(alpha = 0.5, color = 'grey')
plt.title('Pickups over Hour of Day', fontsize = 20)
plt.xlabel('Hour of Day', fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.xticks(rotation=0)
plt.yticks(fontsize = 18)

fig.tight_layout()
# print the total number of Taxi pickups
print ("There were a total of %d Taxi pickups made" % (len(train)))


# **Taxi Vendors**

# In[8]:


# Count number of pickups made per vendor, over month, day of week and hour of day
month_vendor = df.groupby(['month', 'vendor_id']).size().unstack()
dow_vendor = df.groupby(['dow', 'vendor_id']).size().unstack()
hour_vendor = df.groupby(['hour', 'vendor_id']).size().unstack()

# Count vehicles with server access
server = df.groupby(['store_and_fwd_flag', 'vendor_id']).size().unstack()


# In[9]:


# Set custom xtick labels
x_tick_labels_month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
x_tick_labels_day = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(9, 8))

figure = plt.subplot(2, 2, 1)
month_vendor.plot.bar(stacked=True, alpha = 0.7, ax = figure, legend = False)
plt.title('Pickups over Month of Year', fontsize = 13)
plt.xlabel('Month', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.xticks(month_usage.index - 1, x_tick_labels_month, rotation='90', fontsize=12)
plt.xticks(rotation=0)

figure = plt.subplot(2, 2, 2)
dow_vendor.plot.bar(stacked=True, alpha = 0.7, ax = figure, legend = False)
plt.title('Pickups over Day of Week', fontsize = 13)
plt.xlabel('Day of Week', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.xticks(dow_usage.index, x_tick_labels_day, rotation='90', fontsize=12)
plt.xticks(rotation=0)


figure = plt.subplot(2, 2, 3)
hour_vendor.plot.bar(stacked=True, alpha = 0.7, ax = figure, legend = False)
plt.title('Pickups over Hour of Day', fontsize = 13)
plt.xlabel('Hour of Day', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.xticks(rotation=0)

figure = plt.subplot(2, 2, 4)
server.plot.bar(stacked=True, alpha = 0.7, ax = figure)
plt.title('Vehicle Server Access', fontsize = 13)
plt.xlabel(' ', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.xticks(rotation=0)

fig.tight_layout()


# **Multiple Passengers**

# In[10]:


print ("Max passengers at once: %d" % df['passenger_count'].max())
print ("Average passengers at once: %d" % df['passenger_count'].mean())


# In[11]:


# Count number of pickups made per vendor, over month, day of week and hour of day
month_pass = df.groupby(['month', 'passenger_count']).size().unstack()
dow_pass = df.groupby(['dow', 'passenger_count']).size().unstack()
hour_pass = df.groupby(['hour', 'passenger_count']).size().unstack()


# In[12]:


# Set custom xtick labels
x_tick_labels_month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
x_tick_labels_day = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(19, 15))

figure = plt.subplot(2, 2, 1)
month_pass.plot.bar(stacked=True, alpha = 0.8, ax = figure, legend = False)
plt.title('Multi-passenger # over Month of Year', fontsize = 20)
plt.xlabel('Month', fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.xticks(month_usage.index - 1, x_tick_labels_month, rotation='90', fontsize=18)
plt.yticks(fontsize = 18)
plt.xticks(rotation=0)

figure = plt.subplot(2, 2, 2)
dow_pass.plot.bar(stacked=True, alpha = 0.8, ax = figure, legend = False)
plt.title('Multi-passenger # over Day of Week', fontsize = 20)
plt.xlabel('Day of Week', fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.xticks(dow_usage.index, x_tick_labels_day, rotation='90', fontsize=18)
plt.yticks(fontsize = 18)
plt.xticks(rotation=0)


figure = plt.subplot(2, 1, 2)
hour_pass.plot.bar(stacked=True, alpha = 0.8, ax = figure, legend = False)
plt.title('Multi-passenger # over Hour of Day', fontsize = 20)
plt.xlabel('Hour of Day', fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.xticks(rotation = 0, fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(loc = "upper left")


fig.tight_layout()


# In[13]:


train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)


# In[14]:


train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)
plt.hist(train['log_trip_duration'].values, bins=50)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()


# Converting *store_and_fwd_flag *  from {N,Y} to {0,1}

# In[15]:


train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')


# To look where is the most pick up point point in the city.

# In[16]:


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(train['pickup_longitude'].values[:N], train['pickup_latitude'].values[:N],
              color='green', s=1, label='train', alpha=0.1)
ax[1].scatter(test['pickup_longitude'].values[:N], test['pickup_latitude'].values[:N],
              color='red', s=1, label='test', alpha=0.1)
fig.suptitle('Train and test area complete overlap.')
ax[0].legend(loc=0)
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[1].set_xlabel('longitude')
ax[1].legend(loc=0)
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


# In[17]:


coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA().fit(coords)
train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]


# In[18]:


fig, ax = plt.subplots(ncols=2)
ax[0].scatter(train['pickup_longitude'].values[:N], train['pickup_latitude'].values[:N], color='red', s=1, alpha=0.1)
ax[1].scatter(train['pickup_pca0'].values[:N], train['pickup_pca1'].values[:N], color='green', s=1, alpha=0.1)
fig.suptitle('Pickup lat long coords and PCA transformed coords.')
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[1].set_xlabel('pca0')
ax[1].set_ylabel('pca1')
ax[0].set_xlim(city_long_border)
ax[0].set_ylim(city_lat_border)
pca_borders = pca.transform([[x, y] for x in city_lat_border for y in city_long_border])
ax[1].set_xlim(pca_borders[:, 0].min(), pca_borders[:, 0].max())
ax[1].set_ylim(pca_borders[:, 1].min(), pca_borders[:, 1].max())
plt.show()


# **Distance**
# As features in training and testing data is very limited but we can infer some of the Features from Data. 
# The best way should be using a map distance by some Map APi but as the training and test data goes beyond the limit of query limit we will use haversine distance.
# We have to measure Manhattan Distance as well.
# 

# In[19]:


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

train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)


test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)


train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2


# As datatime stamp is always unique we have to extract some features from them like, day og the month, day of the week, hour of the day etc.

# In[20]:


train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday
train.loc[:, 'pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear
train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour
train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute
train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()
train.loc[:, 'pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']

test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday
test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear
test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour
test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute
test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()
test.loc[:, 'pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']


# In[21]:


train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']
fig, ax = plt.subplots(ncols=3, sharey=True)
ax[0].plot(train.groupby('pickup_hour').mean()['avg_speed_h'], 'bo-', lw=2, alpha=0.7)
ax[1].plot(train.groupby('pickup_weekday').mean()['avg_speed_h'], 'go-', lw=2, alpha=0.7)
ax[2].plot(train.groupby('pickup_week_hour').mean()['avg_speed_h'], 'ro-', lw=2, alpha=0.7)
ax[0].set_xlabel('hour')
ax[1].set_xlabel('weekday')
ax[2].set_xlabel('weekhour')
ax[0].set_ylabel('average speed')
fig.suptitle('Rush hour average traffic speed')
plt.show()


# In[22]:


train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 3)
train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 3)
# Average speed for regions
gby_cols = ['pickup_lat_bin', 'pickup_long_bin']
coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
coord_stats = coord_stats[coord_stats['id'] > 100]
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], color='black', s=1, alpha=0.5)
ax.scatter(coord_stats.pickup_long_bin.values, coord_stats.pickup_lat_bin.values, c=coord_stats.avg_speed_h.values,
           cmap='RdYlGn', s=20, alpha=0.5, vmin=1, vmax=8)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.title('Average speed')
plt.show()

train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)
train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 2)
train.loc[:, 'center_lat_bin'] = np.round(train['center_latitude'], 2)
train.loc[:, 'center_long_bin'] = np.round(train['center_longitude'], 2)
train.loc[:, 'pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600))
test.loc[:, 'pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)
test.loc[:, 'pickup_long_bin'] = np.round(test['pickup_longitude'], 2)
test.loc[:, 'center_lat_bin'] = np.round(test['center_latitude'], 2)
test.loc[:, 'center_long_bin'] = np.round(test['center_longitude'], 2)
test.loc[:, 'pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600))


# **Clustering**

# In[23]:


sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])


# In[24]:


train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
t1 = dt.datetime.now()
print('Time till clustering: %i seconds' % (t1 - t0).seconds)


# In[25]:


fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], s=10, lw=0,
           c=train.pickup_cluster[:N].values, cmap='tab20', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# In[26]:


for gby_col in ['pickup_hour', 'pickup_date', 'pickup_dt_bin',
               'pickup_week_hour', 'pickup_cluster', 'dropoff_cluster']:
    gby = train.groupby(gby_col).mean()[['avg_speed_h', 'avg_speed_m', 'log_trip_duration']]
    gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
    train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)
    test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)

for gby_cols in [['center_lat_bin', 'center_long_bin'],
                 ['pickup_hour', 'center_lat_bin', 'center_long_bin'],
                 ['pickup_hour', 'pickup_cluster'],  ['pickup_hour', 'dropoff_cluster'],
                 ['pickup_cluster', 'dropoff_cluster']]:
    coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
    coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
    coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
    coord_stats = coord_stats[coord_stats['id'] > 100]
    coord_stats.columns = gby_cols + ['avg_speed_h_%s' % '_'.join(gby_cols), 'cnt_%s' %  '_'.join(gby_cols)]
    train = pd.merge(train, coord_stats, how='left', on=gby_cols)
    test = pd.merge(test, coord_stats, how='left', on=gby_cols)


# In[27]:


group_freq = '60min'
df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
train.loc[:, 'pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)
test.loc[:, 'pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)

# Count trips over 60min
df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
df_counts['count_60min'] = df_counts.isnull().rolling(group_freq).count()['id']
train = train.merge(df_counts, on='id', how='left')
test = test.merge(df_counts, on='id', how='left')

# Count how many trips are going to each cluster over time
dropoff_counts = df_all     .set_index('pickup_datetime')     .groupby([pd.TimeGrouper(group_freq), 'dropoff_cluster'])     .agg({'id': 'count'})     .reset_index().set_index('pickup_datetime')     .groupby('dropoff_cluster').rolling('240min').mean()     .drop('dropoff_cluster', axis=1)     .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index()     .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})

train['dropoff_cluster_count'] = train[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)
test['dropoff_cluster_count'] = test[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)


# In[28]:


df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
pickup_counts = df_all     .set_index('pickup_datetime')     .groupby([pd.TimeGrouper(group_freq), 'pickup_cluster'])     .agg({'id': 'count'})     .reset_index().set_index('pickup_datetime')     .groupby('pickup_cluster').rolling('240min').mean()     .drop('pickup_cluster', axis=1)     .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index()     .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'pickup_cluster_count'})

train['pickup_cluster_count'] = train[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)
test['pickup_cluster_count'] = test[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)


# In[29]:


fr1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])
fr2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv',
                               usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_street_info = pd.concat((fr1, fr2))
train = train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')
train_street_info.head()


# In[36]:


feature_names = list(train.columns)
print(np.setdiff1d(train.columns, test.columns))
do_not_use_for_training = ['id', 'log_trip_duration', 'dow','pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration',
                           'pickup_date', 'avg_speed_h', 'avg_speed_m', 'pickup_lat_bin', 'pickup_long_bin',
                           'center_lat_bin', 'month','center_long_bin', 'hour', 'pickup_dt_bin','Events', 'Conditions', 'pickup_datetime_group']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
# print(feature_names)
print('We have %i features.' % len(feature_names))
train[feature_names].count()
y = np.log(train['trip_duration'].values + 1)

t1 = dt.datetime.now()
print('Feature extraction time: %i seconds' % (t1 - t0).seconds)


# In[37]:


Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
dtest = xgb.DMatrix(test[feature_names].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


# In[39]:


xgb_pars = {'min_child_weight': 15, 'eta': 0.5, 'colsample_bytree': 0.3, 'max_depth': 10,
            'subsample': 1, 'lambda': 20, 'nthread': -1, 'booster' : 'gbtree', 'silent': 0,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}


# In[40]:


model = xgb.train(xgb_pars, dtrain, 10, watchlist, early_stopping_rounds=100,
                  maximize=False, verbose_eval=2)


# In[41]:


print('Modeling RMSLE %.5f' % model.best_score)
t1 = dt.datetime.now()
print('Training time: %i seconds' % (t1 - t0).seconds)


# In[ ]:


ytest = model.predict(dtest)
#print('Test shape OK.') if test.shape[0] == ytest.shape[0] else print('Oops')
test['trip_duration'] = np.exp(ytest) - 1
test[['id', 'trip_duration']].to_csv('hamzaEnsari_2.csv.gz', index=False, compression='gzip')

