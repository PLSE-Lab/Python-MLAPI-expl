#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list 
# the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[ ]:


#    Get the train data set: train_df
train = pd.read_csv("../input/train.csv")


# In[ ]:


print(train.head())


# In[ ]:


print(train.store_and_fwd_flag.value_counts())


# In[ ]:


print(train.describe())


# In[ ]:


print(train.groupby('store_and_fwd_flag')['pickup_longitude'].describe())


# In[ ]:


print(train.groupby('store_and_fwd_flag')['pickup_latitude'].describe())


# In[ ]:


print(train.groupby('store_and_fwd_flag')['trip_duration'].describe())


# In[ ]:


#Used the extremes of the trip duration of taxi from the real database"""
trip_duration_min = train.groupby('store_and_fwd_flag')['trip_duration'].describe().values[1,3]
trip_duration_max = train.groupby('store_and_fwd_flag')['trip_duration'].describe().values[1,7]

print(trip_duration_min, trip_duration_max)


# In[ ]:


def drop_duration(df):
    df.loc[df.trip_duration < trip_duration_min,'trip_duration'] = np.nan
    value=df.trip_duration.min()
    df.trip_duration.fillna(value=value, inplace=True) 

    df.loc[df.trip_duration > trip_duration_max,'trip_duration'] = np.nan
    value=df.trip_duration.max()
    df.trip_duration.fillna(value=value, inplace=True)
    return


# In[ ]:


drop_duration(train)


# In[ ]:


def conf_int_duration(df):
    """ Drop off the outliers of trip_duration"""
    
    conf_int_duration = np.percentile(df.trip_duration, [2.5,97.5])
    print('\nConfidental interval trip_duration: {}'.format(conf_int_duration))
    
    df.loc[df.trip_duration < conf_int_duration[0],'trip_duration'] = np.nan
    value=df.trip_duration.min()
    df.trip_duration.fillna(value=value, inplace=True) 
    
    df.loc[df.trip_duration > conf_int_duration[1],'trip_duration'] = np.nan
    value=df.trip_duration.max()
    df.trip_duration.fillna(value=value, inplace=True)
    
    print("Trip_duration describe past drop:\n",df.trip_duration.describe())
    
    return


# In[ ]:


conf_int_duration(train)


# In[ ]:


#    Get the test data set: test
test = pd.read_csv("../input/test.csv")
test.info()


# In[ ]:


print(test.groupby('store_and_fwd_flag')['pickup_longitude'].describe())


# To process the same fields, combine two tables

# In[ ]:


result = pd.concat([train, test])
print(result.info())


# In[ ]:


# Used the extremes of the longitude and latitude from the real database - flag "Y"
pickup_long_min = result.groupby('store_and_fwd_flag')['pickup_longitude'].describe().values[1,3]
pickup_long_max = result.groupby('store_and_fwd_flag')['pickup_longitude'].describe().values[1,7]

print("The pickup_longitude min: {}, max: {}".format(pickup_long_min, pickup_long_max))

dropoff_long_min = result.groupby('store_and_fwd_flag')['dropoff_longitude'].describe().values[1,3]
dropoff_long_max = result.groupby('store_and_fwd_flag')['dropoff_longitude'].describe().values[1,7]

print("The dropoff_longitude min: {}, max: {}".format(dropoff_long_min, dropoff_long_max))

pickup_lat_min = result.groupby('store_and_fwd_flag')['pickup_latitude'].describe().values[1,3]
pickup_lat_max = result.groupby('store_and_fwd_flag')['pickup_latitude'].describe().values[1,7]

print("The pickup_latitude min: {}, max: {}".format(pickup_lat_min, pickup_lat_max))

dropoff_lat_min = result.groupby('store_and_fwd_flag')['dropoff_latitude'].describe().values[1,3]
dropoff_lat_max = result.groupby('store_and_fwd_flag')['dropoff_latitude'].describe().values[1,7]

print("The dropoff_latitude min: {}, max: {}".format(dropoff_lat_min, dropoff_lat_max))


# In[ ]:


result['pickup_longitude'] = result.pickup_longitude.round(5)
result['pickup_latitude'] = result.pickup_latitude.round(5)


result.loc[result.pickup_latitude <  pickup_lat_min, 'pickup_latitude'] = np.nan
value=result.pickup_latitude.min()
result.pickup_latitude.fillna(value=value, inplace=True)

result.loc[result.pickup_latitude > pickup_lat_max, 'pickup_latitude'] = np.nan
value=result.pickup_latitude.max()
result.pickup_latitude.fillna(value=value, inplace=True)

result.loc[result.pickup_longitude < pickup_long_min, 'pickup_longitude'] = np.nan
value=result.pickup_longitude.min()
result.pickup_longitude.fillna(value=value, inplace=True)

result.loc[result.pickup_longitude > pickup_long_max, 'pickup_longitude'] = np.nan
value=result.pickup_longitude.max()
result.pickup_longitude.fillna(value=value, inplace=True)


result['dropoff_longitude'] = result.dropoff_longitude.round(5)
result['dropoff_latitude'] = result.dropoff_latitude.round(5)


result.loc[result.dropoff_latitude < dropoff_lat_min, 'dropoff_latitude'] = np.nan
value=result.dropoff_latitude.min()
result.dropoff_latitude.fillna(value=value, inplace=True)

result.loc[result.dropoff_latitude > dropoff_lat_max, 'dropoff_latitude'] = np.nan
value=result.dropoff_latitude.max()
result.dropoff_latitude.fillna(value=value, inplace=True)

result.loc[result.dropoff_longitude < dropoff_long_min, 'dropoff_longitude'] = np.nan
value=result.dropoff_longitude.min()
result.dropoff_longitude.fillna(value=value, inplace=True)

result.loc[result.dropoff_longitude > dropoff_long_max, 'dropoff_longitude'] = np.nan
value=result.dropoff_longitude.max()
result.dropoff_longitude.fillna(value=value, inplace=True)

print(result.describe())


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


print(result[result.trip_distance == 0].describe())


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
result.drop('dropoff_datetime', axis=1, inplace=True)


# In[ ]:


result['pickup_datetime'] = pd.to_datetime(result.pickup_datetime)


# In[ ]:


result['days_in_month'] = result['pickup_datetime'][:].dt.days_in_month


# In[ ]:


result['weekday'] = result['pickup_datetime'].dt.weekday


# In[ ]:


result['hour'] = result['pickup_datetime'][:].dt.hour


# In[ ]:


result['minute'] = result['pickup_datetime'][:].dt.minute


# In[ ]:


result['month'] = result['pickup_datetime'][:].dt.month


# In[ ]:


print(result.info())


# In[ ]:


result.drop(['pickup_datetime', 'store_and_fwd_flag', 'id'], axis=1, inplace=True)


# In[ ]:


print(result.info())


# In[ ]:


#result.to_csv('result.csv')


# In[ ]:


# I cut off the test set with new signs.
test= result[result.trip_duration.isnull()]
test.info()


# In[ ]:


test.drop('trip_duration', axis=1, inplace=True)


# In[ ]:


test.info()


# In[ ]:


train = result[result.trip_duration.notnull()]
train.info()


# In[ ]:


print("train.shape", train.shape, "test shape", test.shape)


# In[ ]:


print(train.head())


# In[ ]:


train.trip_duration = (train['trip_duration']+1).apply(np.log)
trip_duration = train.trip_duration.values


# In[ ]:


trip_duration[:10]


# In[ ]:


train.drop('trip_duration', axis=1, inplace=True)


# In[ ]:


print(test.shape, train.shape, trip_duration.shape)


# In[ ]:


X = train.values


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler().fit(X)


# In[ ]:


X_scaled = scaler.transform(X)


# In[ ]:


y = trip_duration


# In[ ]:


print(X_scaled.shape, y.shape)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=96)


# In[ ]:


print("Shape X_train: {}. Shape y_train: {}. \nShape X_test : {}. Shape y_test : {}".      format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))


# In[ ]:


X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, random_state=144)


# In[ ]:


print("Shape X_train: {}. Shape y_train: {}. \nShape X_val : {}. Shape y_val : {}".      format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))


# In[ ]:


import xgboost as xgb


# In[ ]:


train_xgb  = xgb.DMatrix(X_train, label=y_train)
cv_xgb  = xgb.DMatrix(X_val , label=y_val)
evallist = [(train_xgb, 'train'), (cv_xgb, 'valid')]


# In[ ]:


param = {'max_depth':10,
         'objective':'reg:linear',
         'eta'      :.1,
         'subsample':0.3,
         'lambda '  :6,
         'colsample_bytree ':0.3,
         'colsample_bylevel':1,
         'min_child_weight': 20,
        'nthread': -1}  

model = xgb.train(param, train_xgb, num_boost_round=1000, evals = evallist,
                  early_stopping_rounds=50, maximize=False, 
                  verbose_eval=50)

print("score = {:.5f}, n_boost_round = {}".format(model.best_score, 
                                                  model.best_iteration))


# In[ ]:


X_real_test = test.values


# In[ ]:


X_test_scaled = scaler.transform(X_real_test)


# In[ ]:


test_xgb = xgb.DMatrix(X_test_scaled)
y_pred = model.predict(test_xgb)


# In[ ]:


y_pred[:10]


# In[ ]:


y_pred = np.exp(y_pred[:]) - 1


# In[ ]:


print(y_pred[:10])


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv', index_col=0, header=0)


# In[ ]:


submission.shape


# In[ ]:


y_pred.shape


# In[ ]:


submission.trip_duration = y_pred.round(0)
submission.head(10)


# In[ ]:


submission.describe()


# In[ ]:


submission.to_csv('submission.csv')

