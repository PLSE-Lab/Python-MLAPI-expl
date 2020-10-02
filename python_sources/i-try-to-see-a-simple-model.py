#!/usr/bin/env python
# coding: utf-8

# Hello, friends! I'm trying to see here a simple linear model for predicting.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#%matplotlib
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list 
# the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# Let's see what we have...

# In[ ]:


#    Get the train data set: train_df
train = pd.read_csv("../input/train.csv")
train.info()
train.describe()


# In[ ]:


def conf_int_duration(df):
    """ Drop off the outliers of trip_duration"""
    
    conf_int_duration = np.percentile(df.trip_duration, [5.0,95.0])
    print('\nConfidental interval trip_duration: {}'.format(conf_int_duration))
    
    df.loc[df.trip_duration <= conf_int_duration[0],'trip_duration'] = np.nan
    value=df.trip_duration.min()
    df.trip_duration.fillna(value=value, inplace=True) 
    
    df.loc[df.trip_duration >= conf_int_duration[1],'trip_duration'] = np.nan
    value=df.trip_duration.max()
    df.trip_duration.fillna(value=value, inplace=True)
    
    print("Trip_duration describe past drop:\n",df.trip_duration.describe())
    
    return


# In[ ]:


### Change the data of trip_duration by fuction conf_int_duration(df):
conf_int_duration(train) 


# In[ ]:


print("\n train.info():\n{}".format(train.info()))


# In[ ]:


#    Get the test data set: test_df
test = pd.read_csv("../input/test.csv", index_col='id')
test.info()


# To process the same fields, combine two tables

# In[ ]:


result = pd.concat([train, test])
result.info()


# In[ ]:


result.plot(x='dropoff_latitude', y='dropoff_longitude', 
            kind='scatter', marker='.', alpha=0.5, c='r')


# In[ ]:


result.plot(x='dropoff_latitude', y='dropoff_longitude', 
            xlim=(40.0, 42.0), ylim=(-78.0, -70.0), 
            kind='scatter', marker='.', alpha=0.5, c='b')


# In[ ]:


result.plot(x='dropoff_latitude', y='dropoff_longitude',
            xlim=(40.5, 41.0), ylim=(-74.4, -73.3), 
            kind='scatter', marker='.', alpha=0.3, c='g')


# In[ ]:


result.plot(x='dropoff_latitude', y='dropoff_longitude',
            xlim=(40.55, 40.95), ylim=(-74.2, -73.6), 
            kind='scatter', marker='.', s=.2, alpha=0.5, c='y')


# In[ ]:


def conf_int_coordinates(df):
    df['pickup_longitude'] = df.pickup_longitude.round(4)
    df['pickup_latitude'] = df.pickup_latitude.round(4)
    df.loc[:,['pickup_longitude', 'pickup_latitude']].describe()
    
    ###  Get data coordinates only in confidence interval pickup_Latitude:
    conf_int_latit = np.percentile(df.pickup_latitude, [1, 99])
    print(conf_int_latit)
    
    df.loc[df.pickup_latitude <= conf_int_latit[0], 'pickup_latitude'] = np.nan
    value=df.pickup_latitude.min()
    df.pickup_latitude.fillna(value=value, inplace=True)
    
    df.loc[df.pickup_latitude >= conf_int_latit[1], 'pickup_latitude'] = np.nan
    value=df.pickup_latitude.max()
    df.pickup_latitude.fillna(value=value, inplace=True)
    
    ###  Get data coordinates only in confidence interval pickup_longitude:
    conf_int_longit = np.percentile(df.pickup_longitude,  [1, 99])
    print(conf_int_longit)
    
    df.loc[df.pickup_longitude <= conf_int_longit[0], 'pickup_longitude'] = np.nan
    value=df.pickup_longitude.min()
    df.pickup_longitude.fillna(value=value, inplace=True)
    
    df.loc[df.pickup_longitude >= conf_int_longit[1], 'pickup_longitude'] = np.nan
    value=df.pickup_longitude.max()
    df.pickup_longitude.fillna(value=value, inplace=True)
    
    df['dropoff_longitude'] = df.dropoff_longitude.round(4)
    df['dropoff_latitude'] = df.dropoff_latitude.round(4)
    df.loc[:,['pickup_longitude', 'pickup_latitude']].describe()
    
    ###  Get data coordinates only in confidence interval dropoff_latitude:
    conf_int_latit = np.percentile(df.dropoff_latitude,  [1,99])
    print(conf_int_latit)
    
    df.loc[df.dropoff_latitude <= conf_int_latit[0], 'dropoff_latitude'] = np.nan
    value=df.dropoff_latitude.min()
    df.dropoff_latitude.fillna(value=value, inplace=True)
    
    df.loc[df.dropoff_latitude >= conf_int_latit[1], 'dropoff_latitude'] = np.nan
    value=df.dropoff_latitude.max()
    df.dropoff_latitude.fillna(value=value, inplace=True)
    
    ###  Get data coordinates only in confidence interval dropoff_longitude
    conf_int_longit = np.percentile(df.dropoff_longitude,  [1,99])
    print(conf_int_longit)
    
    df.loc[df.dropoff_longitude <= conf_int_longit[0], 'dropoff_longitude'] = np.nan
    value=df.dropoff_longitude.min()
    df.dropoff_longitude.fillna(value=value, inplace=True)
    
    df.loc[df.dropoff_longitude >= conf_int_longit[1], 'dropoff_longitude'] = np.nan
    value=df.dropoff_longitude.max()
    df.dropoff_longitude.fillna(value=value, inplace=True)
    
    print(df.describe())
    return


# In[ ]:


conf_int_coordinates(result)


# In[ ]:


print("\nresult.info():\n{}".format(result.info()))


# In[ ]:


#	Distance of route
AVG_EARTH_RADIUS = 6371  # in km
def haversine(df, miles=True):
    """ Get the distance of routes by  the haversinus formula"""
    lat1, lng1, lat2, lng2 = (df.pickup_latitude[:], 
                              df.pickup_longitude[:], 
                              df.dropoff_latitude[:], 
                              df.dropoff_longitude[:])
    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat*0.5)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng*0.5)**2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    if miles:
        df['trip_distance'] = h * 0.621371  # in miles
        df['trip_distance'] = df.trip_distance.round(2)
        print(df.trip_distance.describe())
        return 
    else:
        df['trip_distance'] = h  # in kilometers
        df['trip_distance'] = df.trip_distance.round(2)
        print(df.trip_distance.describe())
        return


# In[ ]:


haversine(result, miles=True) 


# In[ ]:


def conf_int_distance(df):
    """ Drop off values of distance by confidantal interval"""
    conf_int_distance = np.percentile(df.trip_distance, [2.5,97.5])
    print('\nConfidental interval trip_duration: {}'.format(conf_int_distance))
    df.loc[df.trip_distance <= conf_int_distance[0],'trip_distance'] = np.nan
    value=df.trip_distance.min() 
    df.trip_distance.fillna(value=value, inplace=True)     
    df.loc[df.trip_distance >= conf_int_distance[1],'trip_distance'] = np.nan
    value=df.trip_distance.max()
    df.trip_distance.fillna(value=value, inplace=True)
    print("\nTrip_distance describe:\n",df.trip_distance.describe())
    return


# In[ ]:


conf_int_distance(result)


# In[ ]:


def arrays_bearing(df):
    """ Get azimuth between points pickup and dropoff"""
    lats1, lngs1, lats2, lngs2 = (df['pickup_latitude'][:], 
                                  df['pickup_longitude'][:], 
                                  df['dropoff_latitude'][:], 
                                  df['dropoff_longitude'][:])
    lats1_rads = np.radians(lats1)
    lats2_rads = np.radians(lats2)
    lngs_delta_rads = np.radians(lngs2 - lngs1)
    
    y = np.sin(lngs_delta_rads) * np.cos(lats2_rads)
    x = np.cos(lats1_rads) * np.sin(lats2_rads) -                          np.sin(lats1_rads) * np.cos(lats2_rads) * np.cos(lngs_delta_rads)
    df['bearing'] = np.degrees(np.arctan2(y, x))
    df['bearing'] = df.bearing.round(0)
    print(df.bearing.describe())
    return


# In[ ]:


arrays_bearing(result)


# In[ ]:


### Drop no useful columns
drop_columns = ['dropoff_datetime', 
                    'dropoff_longitude',
                    'dropoff_latitude']
result.drop(drop_columns, axis=1, inplace=True)


# In[ ]:


def to_dummie_passengers(df):
    """To fix vendor_id, store_and_fwd_flag and passanger_count values"""
    df.loc[df.passenger_count == 0, 'passenger_count'] = np.nan
    value = df.passenger_count.min()
    df.passenger_count.fillna(value=value, inplace=True)
    
    df.loc[df.passenger_count > 6, 'passenger_count'] = np.nan
    value = df.passenger_count.max()
    df.passenger_count.fillna(value=value, inplace=True)
           
    ### Convert to binar number of passengers       
    df['passenger_count'] = df.passenger_count.astype(str)
    df_dummie = pd.get_dummies(df['passenger_count'][:], prefix="pass")
    df_dummie = pd.merge(df[:],df_dummie[:], how='inner', 
                         left_index=True, right_index=True)
    
    return (df_dummie)


# In[ ]:


x = result.loc[:,:]
result = to_dummie_passengers(x)


# In[ ]:


print(result.info())


# In[ ]:


result.drop('passenger_count', axis=1, inplace=True)


# In[ ]:


def to_dummie_vendor(df):    
    ### Convert to binar vendor_id
    df['vendor_id'] = df.vendor_id.astype(str)
    df_dummie = pd.get_dummies(df['vendor_id'][:], prefix="vendor")
    df_dummie = pd.merge(df[:], df_dummie[:], how='inner', 
                         left_index=True, right_index=True)
    
    return(df_dummie)


# In[ ]:


x = result.loc[:,:]
result = to_dummie_vendor(x)


# In[ ]:


print(result.info())


# In[ ]:


result.drop('vendor_id', axis=1, inplace=True)


# In[ ]:


def to_dummie_flag(df):
    ###  Convert to binar flag labels
    df_dummie = pd.get_dummies(df['store_and_fwd_flag'][:], prefix="flag")
    df_dummie = pd.merge(df[:], df_dummie[:], how='inner', 
                         left_index=True, right_index=True)
    
    return (df_dummie)


# In[ ]:


x = result.loc[:,:]
result = to_dummie_flag(x)


# In[ ]:


print(result.info())


# In[ ]:


result.drop('store_and_fwd_flag', axis=1, inplace=True)


# In[ ]:


### Get some clusters of the pickup points
pickup_clusters = np.array(result.loc[:,['pickup_latitude', 'pickup_longitude']])

kmeans = MiniBatchKMeans(n_clusters=16)
kmeans.fit(pickup_clusters)

print('\n Coordinates of cluster centers : {}.       \n Labels of each point : {}.       \n The value of the inertia criterion associated with the chosen partition: {}.       \n The inertia is defined as the sum of square distances of samples       to their nearest neighbor.'.format(kmeans.cluster_centers_,       kmeans.labels_, kmeans.inertia_))


# In[ ]:


sample_len = len(pickup_clusters)
sample_slice = np.random.permutation(sample_len)[:int(sample_len*0.25)]


# In[ ]:


plt.figure(figsize=(8,8))
plt.title = "The clasters of pickup points"
plt.scatter(pickup_clusters[sample_slice,0], 
            pickup_clusters[sample_slice,1],
            c=kmeans.predict(pickup_clusters[sample_slice]), 
            s=.1, alpha=.8, lw=0, cmap='Vega20_r')


# In[ ]:


result['pickup_labels'] = kmeans.labels_
print(result.pickup_labels[:10])


# In[ ]:


drop_columns = ['pickup_latitude', 'pickup_longitude']
result.drop(drop_columns, axis=1, inplace=True)
print(result.info())


# In[ ]:


result['pickup_labels'] = result.pickup_labels.astype(str)
dummies = pd.get_dummies(result['pickup_labels'], prefix="pickup")


# In[ ]:


dummies.head(3)


# In[ ]:


result = pd.merge(result, dummies, how='inner', left_index=True, right_index=True)


# In[ ]:


print(result.info())


# In[ ]:


result.drop('pickup_labels', axis=1, inplace=True)


# In[ ]:


result['pickup_datetime'] = pd.to_datetime(result.pickup_datetime)


# In[ ]:


result['month'] = result['pickup_datetime'][:].dt.month


# In[ ]:


result['days_in_month'] = result['pickup_datetime'][:].dt.days_in_month


# In[ ]:


result['weekday'] = result['pickup_datetime'].dt.weekday_name
      
wdh = result.groupby('weekday')['trip_duration']
(wdh.mean()).plot.hist(bins=25) 


# In[ ]:


df_dummie = pd.get_dummies(result['weekday'][:], prefix="weekday")


# In[ ]:


df_dummie.head(3)


# In[ ]:


result = pd.merge(result[:], df_dummie[:], how='inner', 
                         left_index=True, right_index=True)


# In[ ]:


result.info()


# In[ ]:


result.drop('weekday', axis=1, inplace=True)


# In[ ]:


result['hour'] = result['pickup_datetime'][:].dt.hour


# In[ ]:


result['hour'] = result.hour.astype(str)
df_dummie= pd.get_dummies(result["hour"][:], prefix="hour")
result = pd.merge(result[:], df_dummie[:], how='inner', 
                  left_index=True, right_index=True)


# In[ ]:


result.info()


# In[ ]:


wdh = result.groupby('hour')['trip_duration']
(wdh.mean()).plot.hist(bins=25)


# In[ ]:


result.drop('hour', axis=1, inplace=True)


# In[ ]:


result['minute'] = result['pickup_datetime'][:].dt.minute


# In[ ]:


print(result.info())


# In[ ]:


result.drop('pickup_datetime', axis=1, inplace=True)


# In[ ]:


print(result.info())


# In[ ]:


result.to_csv('result.csv')


# In[ ]:


result.trip_duration.isnull().value_counts()


# In[ ]:


# I cut off the test set with new signs.
test = result[result.trip_duration.isnull()]
test.describe()


# And training set too ...

# In[ ]:


train = result[result.trip_duration.notnull()]
train.describe()


# In[ ]:


print("train.shape", train.shape, "test shape", test.shape)


# In[ ]:


del(result)
del(x)


# In[ ]:


train["trip_duration"] = (train.trip_duration[:] +1).apply(np.log)


# In[ ]:


y = train['trip_duration'][:].values


# In[ ]:


train.drop(['trip_duration', 'id'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


X = train.values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


print("Shape X_train: {}. Shape y_train: {}. \nShape X_test : {}. Shape y_test : {}".      format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))


# In[ ]:


scaler = MinMaxScaler().fit(X_train)


# In[ ]:


X_train_scaled = scaler.transform(X_train)


# In[ ]:


print("X_train[:3]:\n{},\nX_train_scaled[:3]\n{}".          format(X_train[:3], X_train_scaled[:3]))


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(X_train_scaled, y_train)
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("\nAccuracy by train-scaled set: {:.5f}".         format(lr.score(X_train_scaled, y_train)))


# In[ ]:


X_test_scaled = scaler.transform(X_test)
print("\nAccuracy by test set: {:.5f}".         format(lr.score(X_test_scaled, y_test)))


# In[ ]:


ridge = Ridge()
ridge.fit(X_train_scaled, y_train)
print("\nAccuracy by train-scaled set: {:.5f}".      format(ridge.score(X_train_scaled, y_train)))

print("\nAccuracy by test set: {:.5f}".      format(ridge.score(X_test_scaled, y_test)))


# In[ ]:


ridge100 = Ridge(alpha=100).fit(X_train_scaled, y_train)
print("\nAccuracy by train-scaled set: {:.5f}".      format(ridge100.score(X_train_scaled, y_train)))

print("\nAccuracy by test set: {:.5f}".      format(ridge100.score(X_test_scaled, y_test)))


# In[ ]:


ridge001 = Ridge(alpha=0.01).fit(X_train_scaled, y_train)
print("\nAccuracy by train-scaled set: {:.5f}".      format(ridge001.score(X_train_scaled, y_train)))

print("\nAccuracy by test set: {:.5f}".      format(ridge001.score(X_test_scaled, y_test)))


# In[ ]:


plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()


# We can see that the more general picture of accuracy by linear regression, it is smooth and total.

# In[ ]:


X_train = X
y_train = y


# In[ ]:


scaler = MinMaxScaler().fit(X_train)


# In[ ]:


X_train_scaled = scaler.transform(X_train)


# In[ ]:


lr.fit(X_train_scaled, y_train)
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("\nAccuracy by train-scaled set: {:.5f}".         format(lr.score(X_train_scaled, y_train)))


# In[ ]:


X_test = test.drop(['trip_duration', 'id'], axis=1).values


# In[ ]:


print(X_test.shape)


# In[ ]:


X_test_scaled = scaler.transform(X_test)


# In[ ]:


y_pred = lr.predict(X_test_scaled)


# In[ ]:


y_pred = np.exp(y_pred[:]) - 1


# In[ ]:


print(y_pred)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col=0, header=0)


# In[ ]:


submission.head()


# In[ ]:


submission.shape


# In[ ]:


y_pred.shape


# In[ ]:


submission.trip_duration = y_pred
submission.head(20)


# In[ ]:


submission.describe()


# In[ ]:


submission.to_csv('submission.csv')

