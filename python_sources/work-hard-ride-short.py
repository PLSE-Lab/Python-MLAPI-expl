#!/usr/bin/env python
# coding: utf-8

# **ABOUT THE COMPETITION**
# 
# The competition is about predicting the new york taxi trip duration when provided with training features. The training dataset includes 1400000 rows and testing is done on 600000 rows.
# 
# To know which route is the best one to take, we need to be able to predict how long the trip will last when taking a specific route. Therefore, *the goal of this playground competition is to predict the the duration of each trip in the test data set, given start and end coordinates.*

# **OVERVIEW AND INTRODUCTION**
# 
# 
# This kernel is written and developed using IPython Notebook and XGBoost, with the assist of mini-batch Kmeans clustering algorithm. The work flow of the kernel has the following steps-:
# 
# 1. Importing the Sklearn libraries 
# 2. Importing the dataset for the competition ( We have also added another dataset for improvement of the training model by adding more parameters for better prediction)
# 1. Analysis of data such as finding mean, standard deviation, quartiles of each feature. 
# 1. The next step includes cleaning of data to do away eith outliers, since the major part of the algorithm involves using mini batch Kmeans which is sensitive to outliers, the cleaning of data becomes very necessary.
# 1. Then we begin observing our cleaned working data with the help of matplotlib. The plots help us finding the trends and correlation in the data.
# 1. Next we compute the haversine, manhattan and bearing distance which helps us form clusters on the basis of distance between the corresponding pick up and drop off points.
# 1. Next we apply mini-batch Kmeans algorithm to clsuter points on basis of pick-up latitude, pick-up longitude, drop off latitude and drop off latitude.
# 1. The steps that follow are all about feature extraction where average speed haversine, average speed manhattan is computed which further adds to the features for the training set.
# 1. The clustering helps us find cluster centres which are used as separate features for the training set. The clusters formed in the mini batch Kmeans was on the basis of the pick up and drop off locations for which 100 clusters were formed. As a result we get addition 200 features in form of cluster centres ( 100 pick up and 100 drop off clusters)
# 1. Finally, the redundant columns are removed and the back-bone of the kernel which is XGBoost is applied to the dataset with the added parameters and the results are observed.

# **1) IMPORTING NECESSARY SKLEARN LIBRARIES**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, asin, sqrt
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]


# **2) IMPORTING THE TRAINING AND TESTING DATASET**

# In[ ]:


train = pd.read_csv('../input/new-york-city-taxi-with-osrm/train.csv')
test = pd.read_csv('../input/new-york-city-taxi-with-osrm/test.csv')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
train.head()


# **3) DATA ANALYSIS**

# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
train.describe()


# **4) CLEANING OF DATA**
# 
# Here, we apply the standard data cleaning method where the values that lie only between the range m-2*s < x < m+2*s are kept as training points and rest are removed as outliers.
# 
# The feature chosen for the task is trip_duration which makes perfect sense because we eliminate the outliers on the basis of the feature which is supposed to predicted in the task.

# In[ ]:



m = np.mean(train['trip_duration'])
s = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= m + 2*s]
train = train[train['trip_duration'] >= m - 2*s]

train.head()


# Further cleaning of data which includes forming a map with limits provided of pick up and drop off coordinates.

# In[ ]:


train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]


# As a final step in preparing our data we need to change the formatting of the date variables (`pickup_datetime` and `dropoff_datetime`). This will help a lot with data extraction in the coming section.

# In[ ]:


train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime) #Not in Test


#  **5) DATA ANALYSIS AND OBSERVATION**
# 
# These next steps involve looking at the data visually. Often you'll discover looking at something significant as a graph rather than a table (for example) will give you far greater insight into its nature.

# In[ ]:


plt.hist(train['trip_duration'].values, bins=100)
plt.xlabel('trip_duration')
plt.ylabel('number of train records')
plt.show()


# We plot a simple histogram of the trip duration, throwing the data into 100 bins. binning involves taking your data's max and min points, subtracting it to get the length, dividing that length by the number of bins to get the interval length, and grouping the data points into those intervals.

# In[ ]:


train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)
plt.hist(train['log_trip_duration'].values, bins=100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()
sns.distplot(train["log_trip_duration"], bins =100)


# It is very important for us to find whether or not the training and testing data are in agreement with each other or not. By this we mean that we need to find using Time series graph that how well are the number of trips over time varying with the training and testing dataset.
# 
# We'll simply plot a timeseries line graph of both the test and training data to not only look into identifying possible trends but to see if both data sets follow the same pattern shape.

# In[ ]:


plt.plot(train.groupby('pickup_date').count()[['id']], 'o-', label='train')
plt.plot(test.groupby('pickup_date').count()[['id']], 'o-', label='test')
plt.title('Trips over Time.')
plt.legend(loc=0)
plt.ylabel('Trips')
plt.show()


# The following 2 plots tell us that whether or not there's much of a difference between travel times for the two vendors.  There's another feature we can use to see if after all there is a significant difference in mean travel time: the `store_and_fwd_flag`.
# 
# As the results suggest that the vendors ids or the store and forward flag values have similar results even on such a large dataset thereby showing that the trip duration isn't effected by the vendor id's or the store and forward flag value.
# 
# It also makes sense because trip duration between places does not depend on the vendor used for the task. Even in real world scenario OLA and UBER will deliver same travel time when travelling between same points and at the same time.

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
plot_vendor = train.groupby('vendor_id')['trip_duration'].mean()
plt.subplots(1,1,figsize=(17,10))
plt.ylim(ymin=800)
plt.ylim(ymax=820)
sns.barplot(plot_vendor.index,plot_vendor.values)
plt.title('Time per Vendor')
plt.legend(loc=0)
plt.ylabel('Time in Seconds')
plt.show()


# In[ ]:


snwflag = train.groupby('store_and_fwd_flag')['trip_duration'].mean()

plt.subplots(1,1,figsize=(17,10))
plt.ylim(ymin=0)
plt.ylim(ymax=900)
plt.title('Time per store_and_fwd_flag')
plt.legend(loc=0)
plt.ylabel('Time in Seconds')
sns.barplot(snwflag.index,snwflag.values)


# The following plot shows whether or not the passenger count has significant impact on the trip duration.

# In[ ]:


pc = train.groupby('passenger_count')['trip_duration'].mean()

plt.subplots(1,1,figsize=(17,10))
plt.ylim(ymin=0)
plt.ylim(ymax=1100)
plt.title('Time per store_and_fwd_flag')
plt.legend(loc=0)
plt.ylabel('Time in Seconds')
sns.barplot(pc.index,pc.values)


# In[ ]:


train.groupby('passenger_count').size()


# In[ ]:


test.groupby('passenger_count').size()


# 

# We utilise the city map border coordinates for New York, mentioned earlier in the kernel to create the canvas wherein the coordinate points will be graphed. To display the actual coordinates a simple scatter plot is used. It shows whether or not the pick up points in the training and testing dataset overlap is some manner or not. 

# In[ ]:


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(train['pickup_longitude'].values[:100000], train['pickup_latitude'].values[:100000],
              color='blue', s=1, label='train', alpha=0.1)
ax[1].scatter(test['pickup_longitude'].values[:100000], test['pickup_latitude'].values[:100000],
              color='green', s=1, label='test', alpha=0.1)
fig.suptitle('Train and test area complete overlap.')
ax[0].legend(loc=0)
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[1].set_xlabel('longitude')
ax[1].legend(loc=0)
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


# The same is done for the drop off points too. A sense check

# In[ ]:


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(train['dropoff_longitude'].values[:100000], train['dropoff_latitude'].values[:100000],
              color='blue', s=1, label='train', alpha=0.1)
ax[1].scatter(test['dropoff_longitude'].values[:100000], test['dropoff_latitude'].values[:100000],
              color='green', s=1, label='test', alpha=0.1)
fig.suptitle('Train and test area complete overlap.')
ax[0].legend(loc=0)
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[1].set_xlabel('longitude')
ax[1].legend(loc=0)
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


# **6) COMPUTING HAVERSINE AND MANHATTAN DISTANCE**
# 
# We can determine the distance and direction of a specific trip based on the pickup and dropoff coordinates. Applying these functions to both the test and train data, we can calculate the haversine distance which is the great-circle distance between two points on a sphere given their longitudes and latitudes. We can then calculate the summed distance travelled in Manhattan. 

# In[ ]:


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


# Increasing the features by adding addition columns such as distance_haversine, distance_manhattan and direction of travel

# In[ ]:


train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)    
    
train.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)


train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)


# 
# There are three steps to preparing the data: create the coordinates stack, configure the KMeans clustering parameters, and create the actual clusters:

# In[ ]:


coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))


# **7) APPLYING MINI BATCK KMEANS ALGORITHM FOR FURTHER FEATURE EXTRACTION**
# 
# The algorithm works on 800000 datapoints and creates 100 clusters which means that we can further add 100 columns for both pick up location and drop off location. 
# Mini-batches of 10000 are used which further helps in better clustering. We can always play around with this and find which mini batch fetches the best possible result.

# In[ ]:


sample_ind = np.random.permutation(len(coords))[:800000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])


# Feature extraction

# In[ ]:


train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])


# Displaying the clusters

# In[ ]:


fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(train.pickup_longitude.values[:500000], train.pickup_latitude.values[:500000], s=10, lw=0,
           c=train.pickup_cluster[:500000].values, cmap='autumn', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# The following steps include further features extraction such as month, day of the month, hour of the day etc.

# In[ ]:


#Extracting Month
train['Month'] = train['pickup_datetime'].dt.month
test['Month'] = test['pickup_datetime'].dt.month


# In[ ]:


train.groupby('Month').size(),test.groupby('Month').size()


# In[ ]:


train['DayofMonth'] = train['pickup_datetime'].dt.day
test['DayofMonth'] = test['pickup_datetime'].dt.day
len(train.groupby('DayofMonth').size()),len(test.groupby('DayofMonth').size())


# In[ ]:


train['Hour'] = train['pickup_datetime'].dt.hour
test['Hour'] = test['pickup_datetime'].dt.hour
len(train.groupby('Hour').size()),len(test.groupby('Hour').size())


# In[ ]:


train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
test['dayofweek'] = test['pickup_datetime'].dt.dayofweek
len(train.groupby('dayofweek').size()),len(test.groupby('dayofweek').size())


# We can safely use the different date parts in their extracted forms as part of the modelling process. We now take a look at the average speed and how it changes over time, specifically focusing on how the hour of the day, the day of the week, and the month of the year and how it affects average speed.

# Further plots for better understanding.

# In[ ]:


train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']
fig, ax = plt.subplots(ncols=3, sharey=True)
ax[0].plot(train.groupby('Hour').mean()['avg_speed_h'], 'bo-', lw=2, alpha=0.7)
ax[1].plot(train.groupby('dayofweek').mean()['avg_speed_h'], 'go-', lw=2, alpha=0.7)
ax[2].plot(train.groupby('Month').mean()['avg_speed_h'], 'ro-', lw=2, alpha=0.7)
ax[0].set_xlabel('Hour of Day')
ax[1].set_xlabel('Day of Week')
ax[2].set_xlabel('Month of Year')
ax[0].set_ylabel('Average Speed')
fig.suptitle('Average Traffic Speed by Date-part')
plt.show()


# In[ ]:


train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 3)
train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 3)
# Average speed for regions
gby_cols = ['pickup_lat_bin', 'pickup_long_bin']
coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
coord_stats = coord_stats[coord_stats['id'] > 100]
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(train.pickup_longitude.values[:500000], train.pickup_latitude.values[:500000], color='black', s=1, alpha=0.5)
ax.scatter(coord_stats.pickup_long_bin.values, coord_stats.pickup_lat_bin.values, c=coord_stats.avg_speed_h.values,
           cmap='RdYlGn', s=20, alpha=0.5, vmin=1, vmax=8)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.title('Average speed')
plt.show()


# Traffic usually peaks between 5am and 9am, and then again from about 4pm to around 6 or 7pm. But it would seem in manhattan that average speed diminishes as the day goes by from around 6am and picks up again around 7 or 8pm. So most of the travelling during work hours. The average speed by weekday follows an expected trend. Over the weekend (Friday, Saturday, Sunday) the average speed picks up quite nicely, indicating less traffic. Finally, the average trip speed by month follows an expected trend. In the winter months there are less trips indicating less traffic in general in the city which means you can average a higher speed on the roads.
# These plots therefore help us in understanding the importance of the features such as time of the day, month day of the week etc.
# 

# **8) FEATURE EXTRACTION AND DATA ENRICHMENT**
# 
# 
# We import the OSRM dataset for further features and fastest routes between given datapoints. Ultimately this is a data set containing the fastest routes from specific starting points in New York.
# 
# 

# In[ ]:


fr1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])
fr2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv',
                               usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_street_info = pd.concat((fr1, fr2))
train = train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')


# Creating dummy variables for each feature and extracting columns from each possible value of each possible feature.
# 
# For example if passenger count has 9 possible values then the passenger_count feature alone fetches us 9 separate features that correspond with the 9 possible values of that feature.

# In[ ]:


vendor_train = pd.get_dummies(train['vendor_id'], prefix='vi', prefix_sep='_')
vendor_test = pd.get_dummies(test['vendor_id'], prefix='vi', prefix_sep='_')
passenger_count_train = pd.get_dummies(train['passenger_count'], prefix='pc', prefix_sep='_')
passenger_count_test = pd.get_dummies(test['passenger_count'], prefix='pc', prefix_sep='_')
store_and_fwd_flag_train = pd.get_dummies(train['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
store_and_fwd_flag_test = pd.get_dummies(test['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
cluster_pickup_train = pd.get_dummies(train['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_pickup_test = pd.get_dummies(test['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_dropoff_train = pd.get_dummies(train['dropoff_cluster'], prefix='d', prefix_sep='_')
cluster_dropoff_test = pd.get_dummies(test['dropoff_cluster'], prefix='d', prefix_sep='_')

month_train = pd.get_dummies(train['Month'], prefix='m', prefix_sep='_')
month_test = pd.get_dummies(test['Month'], prefix='m', prefix_sep='_')
dom_train = pd.get_dummies(train['DayofMonth'], prefix='dom', prefix_sep='_')
dom_test = pd.get_dummies(test['DayofMonth'], prefix='dom', prefix_sep='_')
hour_train = pd.get_dummies(train['Hour'], prefix='h', prefix_sep='_')
hour_test = pd.get_dummies(test['Hour'], prefix='h', prefix_sep='_')
dow_train = pd.get_dummies(train['dayofweek'], prefix='dow', prefix_sep='_')
dow_test = pd.get_dummies(test['dayofweek'], prefix='dow', prefix_sep='_')


# Finding the shape of added features. Asimple sense check to find whether or not we are on the right track.

# In[ ]:


vendor_train.shape,vendor_test.shape


# In[ ]:


passenger_count_train.shape,passenger_count_test.shape


# In[ ]:


cluster_pickup_train.shape,cluster_pickup_test.shape


# In[ ]:


cluster_dropoff_train.shape,cluster_dropoff_test.shape


# In[ ]:


dom_train.shape,dom_test.shape


# In[ ]:


hour_train.shape,hour_test.shape


# In[ ]:


dow_train.shape,dow_test.shape


# In[ ]:


passenger_count_test = passenger_count_test.drop('pc_9', axis = 1)


# Now that we have added external features and extracted all the possible values of all the possible features, it becomes important to do away with the redundant columns and the next step involves removing all the redundant features. An important point to notice is that we also do away with the average distance feature because it is still a function of distance which is a feature in itself. 

# In[ ]:


train = train.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','Month','DayofMonth','Hour','dayofweek',
                   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis = 1)
Test_id = test['id']
test = test.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','Month','DayofMonth','Hour','dayofweek',
                   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis = 1)

train = train.drop(['dropoff_datetime','avg_speed_h','avg_speed_m','pickup_lat_bin','pickup_long_bin','trip_duration'], axis = 1)


# In[ ]:


train.shape,test.shape


# **9) THE FINAL MATRIX CONTAINING ALL THE IMPORTANT FEATURES BY CONCATENATING ALL THE FEATURES EXTRACTED**

# In[ ]:


Train_Master = pd.concat([train,
                          vendor_train,
                          passenger_count_train,
                          store_and_fwd_flag_train,
                          cluster_pickup_train,
                          cluster_dropoff_train,
                         month_train,
                         dom_train,
                          hour_test,
                          dow_train
                         ], axis=1)


# In[ ]:


Test_master = pd.concat([test, 
                         vendor_test,
                         passenger_count_test,
                         store_and_fwd_flag_test,
                         cluster_pickup_test,
                         cluster_dropoff_test,
                         month_test,
                         dom_test,
                          hour_test,
                          dow_test], axis=1)


# In[ ]:


Train_Master.shape,Test_master.shape


# In[ ]:


Train_Master = Train_Master.drop(['pickup_datetime','pickup_date'],axis = 1)
Test_master = Test_master.drop(['pickup_datetime','pickup_date'],axis = 1)


# Train test split is such that 80 % dataset for traning and remaining 20% for testing i.e. validation.

# In[ ]:


Train, Test = train_test_split(Train_Master[0:800000], test_size = 0.2)


# In[ ]:


X_train = Train.drop(['log_trip_duration'], axis=1)
Y_train = Train["log_trip_duration"]
X_test = Test.drop(['log_trip_duration'], axis=1)
Y_test = Test["log_trip_duration"]

Y_test = Y_test.reset_index().drop('index',axis = 1)
Y_train = Y_train.reset_index().drop('index',axis = 1)


# **10) THE FINAL STEP WHICH INVOLVES FORMING THE XGBOOST MATRIX AND PREDICTION**

# In[ ]:


dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_test, label=Y_test)
dtest = xgb.DMatrix(Test_master)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


# XGBoost algorithm with the parameters. We can play around with the parameters but before that we must study about XGBoost because it's documentation helps greatly in understanding how to fine tune the parameters for better performance and analysis result.
# 
# The features included are-:
# 
# 1. max depth = 6
# 1. learning rate = 0.09
# 1. iteration = 250 

# In[ ]:


xgb_pars = {'min_child_weight': 1, 'eta': 0.09, 'colsample_bytree': 0.9, 
            'max_depth': 6,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 250, watchlist, early_stopping_rounds=250,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)


#  Feature importance graph

# In[ ]:


xgb.plot_importance(model, max_num_features=28, height=0.7)


# Prediction and final submission.

# In[ ]:


pred = model.predict(dtest)
pred = np.exp(pred) - 1


# In[ ]:


submission = pd.concat([Test_id, pd.DataFrame(pred)], axis=1)
submission.columns = ['id','trip_duration']
submission['trip_duration'] = submission.apply(lambda x : 1 if (x['trip_duration'] <= 0) else x['trip_duration'], axis = 1)
submission.to_csv("submission.csv", index=False)


# In[ ]:





# In[ ]:




