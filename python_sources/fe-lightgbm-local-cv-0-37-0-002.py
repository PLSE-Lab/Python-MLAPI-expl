#!/usr/bin/env python
# coding: utf-8

# Feature engineering and LightGBM framework.
# Datasets:
# 
#  - New York City Taxi Trip Duration (https://www.kaggle.com/c/nyc-taxi-trip-duration)
#  - New York City Taxi with OSRM (https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm)
#  - Weather data in New York City - 2016 (https://www.kaggle.com/mathijs/weather-data-in-new-york-city-2016)
#  - New York City Taxi Trip - Hourly Weather Data (https://www.kaggle.com/meinertsen/new-york-city-taxi-trip-hourly-weather-data)

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Plots
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import gc 
# LightGBM framework
import lightgbm as lgb


# ## Data

# In[2]:


### Get the data
# Main dataset
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
# New York City Taxi with OSRM 
# (https://www.kaggle.com/mathijs/weather-data-in-new-york-city-2016)
train_fastest_1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv')
train_fastest_2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv')
test_fastest = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv')
# Weather data in New York City - 2016
# (https://www.kaggle.com/mathijs/weather-data-in-new-york-city-2016)
weather = pd.read_csv('../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016.csv')
# New York City Taxi Trip - Hourly Weather Data
# (https://www.kaggle.com/meinertsen/new-york-city-taxi-trip-hourly-weather-data)
weather_hour = pd.read_csv('../input/new-york-city-taxi-trip-hourly-weather-data/Weather.csv')
# Train-validation split
train, valid, _, _ = train_test_split(train, train.trip_duration, 
                                      test_size=0.2, random_state=2017)
# Add set marker
train['eval_set'] = 0; valid['eval_set'] = 1; test['eval_set'] = 2
test['trip_duration'] = np.nan; test['dropoff_datetime'] = np.nan
# Glue tables
frame = pd.concat([train, valid, test], axis=0)
frame_fastest = pd.concat([train_fastest_1, train_fastest_2, test_fastest], axis = 0)

### Memory optimization
# Main dataframe
frame.eval_set = frame.eval_set.astype(np.uint8)
frame.passenger_count = frame.passenger_count.astype(np.int8)
frame.store_and_fwd_flag = pd.get_dummies(frame['store_and_fwd_flag'], 
                                          prefix='store_and_fwd_flag', drop_first=True)
frame.vendor_id = frame.vendor_id.astype(np.int8)
# Weather dataframe
weather.replace('T', 0.001, inplace=True)
weather['date'] = pd.to_datetime(weather['date'], dayfirst=True).dt.date
weather['average temperature'] = weather['average temperature'].astype(np.int64)
weather['precipitation'] = weather['precipitation'].astype(np.float64)
weather['snow fall'] = weather['snow fall'].astype(np.float64)
weather['snow depth'] = weather['snow depth'].astype(np.float64)
# Weather hourly dataframe
weather_hour['Datetime'] = pd.to_datetime(weather_hour['pickup_datetime'], dayfirst=True)
weather_hour['date'] = weather_hour.Datetime.dt.date
weather_hour['hour'] = weather_hour.Datetime.dt.hour
weather_hour['hour'] = weather_hour.hour.astype(np.int8)
weather_hour['fog'] = weather_hour.fog.astype(np.int8)
weather_hour = weather_hour[['date', 'hour', 'tempm', 'dewptm', 'hum', 'wspdm', 
                             'wdird', 'vism', 'pressurei', 'fog']]
del train, valid, test, train_fastest_1, train_fastest_2, test_fastest
gc.collect();


# ## Adding features

# In[3]:


# Define clusters
def clusters(df, pic=False):
    coords = np.vstack((df[['pickup_longitude', 'pickup_latitude']].values,
                        df[['dropoff_longitude', 'dropoff_latitude']].values))
    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
    cl_pickup = kmeans.predict(df[['pickup_longitude', 'pickup_latitude']])
    cl_dropoff = kmeans.predict(df[['dropoff_longitude', 'dropoff_latitude']])
    # If pic = True, show pictures
    if pic:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.scatter(df.pickup_longitude.values, 
                    df.pickup_latitude.values, 
                    s=0.2, lw=0, c=cl_pickup, cmap='tab20', alpha=0.5)
        plt.xlim(-74.03, -73.77); plt.ylim(40.63, 40.85)
        plt.xlabel('pickup_longitude'); plt.ylabel('pickup_latitude')
        plt.subplot(1,2,2)
        plt.scatter(df.dropoff_longitude.values, 
                    df.dropoff_latitude.values, 
                    s=0.2, lw=0, c=cl_dropoff, cmap='tab20', alpha=0.5)
        plt.xlim(-74.03, -73.77); plt.ylim(40.63, 40.85)
        plt.xlabel('dropoff_longitude'); plt.ylabel('dropoff_latitude')
        plt.show()
    return cl_pickup, cl_dropoff

# Rotate the map
def rotate_coords(df, col1, col2, pic=False):
    alpha = 0.610865 # angle = 35 degrees
    #alpha = 0.506145 # angle = 29 degrees
    # Center of rotation
    x_c = df[col1].mean()
    y_c = df[col2].mean()
    # Coordinates
    C = df[[col1, col2]] - np.array([x_c, y_c])
    # Rotation matrix
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha),  np.cos(alpha)]])
    C_rot = np.matmul(R, C.transpose().values).transpose() + np.array([x_c, y_c])    
    if pic:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.scatter(df[col1], df[col2], s=0.2, alpha=0.5)
        plt.xlabel(col1); plt.ylabel(col2); 
        plt.ylim(40.6, 40.9); plt.xlim(-74.1, -73.7);
        plt.subplot(1,2,2)
        plt.scatter(C_rot[:, 0], C_rot[:, 1], s=0.2, alpha=0.5)
        plt.xlabel(col1+'_rot'); plt.ylabel(col2+'_rot'); 
        plt.ylim(40.6, 40.9); plt.xlim(-74.1, -73.7);
        plt.show()
    return C_rot

# Manhattan distances
def my_manhattan_distances(x1, x2, y1, y2):
    return np.abs(x1 - x2) + np.abs(y1 - y2)
# Euclidean distances
def my_euclidean_distances(x1, x2, y1, y2):
    return np.square(x1 - x2) + np.square(y1 - y2)
my_manhattan_distances = np.vectorize(my_manhattan_distances)
my_euclidean_distances = np.vectorize(my_euclidean_distances)

# Adding features
def add_features(df, predict=False):
    # If predict = True, this function will prepare (add new features) 
    # train set (all train data) and test (all test data); else, 
    # if predict = False, the function will prepare train and validation datasets
    if predict: 
        train_inds = df[(df.eval_set != 2)].index
    else:
        df = df[(df.eval_set != 2)].copy()
        train_inds = df[df.eval_set != 1].index
    
    ### Log trip
    print('Log trip duration')
    df['trip_duration'] = df.trip_duration.apply(np.log)
    
    ### PCA transformation
    print('Add PCA geo-coordinates')
    coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']], 
                        df[['dropoff_latitude', 'dropoff_longitude']]))
    pca = PCA().fit(coords) # define 2 main axis
    df['pickup_pca0'] = pca.transform(df[['pickup_longitude', 'pickup_latitude']])[:,0]
    df['pickup_pca1'] = pca.transform(df[['pickup_longitude', 'pickup_latitude']])[:,1]
    df['dropoff_pca0'] = pca.transform(df[['dropoff_longitude', 'dropoff_latitude']])[:,0]
    df['dropoff_pca1'] = pca.transform(df[['dropoff_longitude', 'dropoff_latitude']])[:,1]
    df['distance_pca0'] = np.abs(df.pickup_pca0-df.dropoff_pca0)
    df['distance_pca1'] = np.abs(df.pickup_pca1-df.dropoff_pca1)
    
    print('Rorate geo-coordinates')
    C_rot_pickup = rotate_coords(df, 'pickup_longitude', 'pickup_latitude', not predict)
    C_rot_dropoff = rotate_coords(df, 'dropoff_longitude', 'dropoff_latitude', not predict)
    df['pickup_longitude_rot'] = C_rot_pickup[:, 0]
    df['pickup_latitude_rot'] = C_rot_pickup[:, 1]
    df['dropoff_longitude_rot'] = C_rot_dropoff[:, 0]
    df['dropoff_latitude_rot'] = C_rot_dropoff[:, 1]
    
    ### Add clusters
    print('Add clusters')
    cl_pu, cl_do = clusters(df, not predict)
    df['pickup_clusters'] = cl_pu
    df['dropoff_clusters'] = cl_do
       
    ### to DateTime
    print('Convert to datetime format')
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    # Add weather info
    df['date'] = df['pickup_datetime'].dt.date # adding date column
    df['hour'] = df['pickup_datetime'].dt.hour # adding hour column
    df = pd.merge(left=df, right=weather, on='date', how='left')
    # Add weather hourly
    df = pd.merge(left=df, right=weather_hour.drop_duplicates(subset=['date', 'hour']), 
                  on=['date', 'hour'], how='left')
    df.drop(['date'], axis=1, inplace=True)
    # Weather added
    df['month'] = df['pickup_datetime'].dt.month
    df['week_of_year'] = df['pickup_datetime'].dt.week
    df['day'] = df['pickup_datetime'].dt.day
    df['month_day'] = df['month'] + df['day']
    df['day_of_year'] = df['pickup_datetime'].dt.dayofyear
    #df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_year_hour'] = 24*df['day_of_year'] + df['hour']
    df['hour_minute'] = 60*df['hour'] + df['pickup_datetime'].dt.minute
    df['day_week'] = df['pickup_datetime'].dt.weekday
    df['month_day_hour'] = 31*24*df['month'] + 24*df['day'] + df['hour']
    
    ### Some usfull averages ###
    print('Add averages')
    train_info = df.iloc[train_inds].copy() # get only train data
    # Month average 
    month_avg = train_info.groupby('month').trip_duration.mean()
    month_avg = month_avg.reset_index(); month_avg.columns = ['month', 'month_avg']
    df = pd.merge(left=df, right=month_avg, on='month', how='left')
    
    
    # Week of year average
    week_year_avg = train_info.groupby('week_of_year').trip_duration.mean()
    week_year_avg = week_year_avg.reset_index()
    week_year_avg.columns = ['week_of_year', 'week_of_year_avg']
    df = pd.merge(left=df, right=week_year_avg, on='week_of_year', how='left')
        
    # Day of month average
    day_month_avg = train_info.groupby('day').trip_duration.mean()
    day_month_avg = day_month_avg.reset_index()
    day_month_avg.columns = ['day', 'day_of_month_avg']
    df = pd.merge(left=df, right=day_month_avg, on='day', how='left')
        
    # Day of year average
    day_year_avg = train_info.groupby('day_of_year').trip_duration.mean()
    day_year_avg = day_year_avg.reset_index()
    day_year_avg.columns = ['day_of_year', 'day_of_year_avg']
    df = pd.merge(left=df, right=day_year_avg, on='day_of_year', how='left')
        
    # Hour average
    hour_avg = train_info.groupby('hour').trip_duration.mean()
    hour_avg = hour_avg.reset_index(); hour_avg.columns = ['hour', 'hour_avg']
    df = pd.merge(left=df, right=hour_avg, on='hour', how='left')
        
    # Day week average
    day_week_avg = train_info.groupby('day_week').trip_duration.mean()
    day_week_avg = day_week_avg.reset_index()
    day_week_avg.columns = ['day_week', 'day_week_avg']
    df = pd.merge(left=df, right=day_week_avg, on='day_week', how='left')  
    
    # Clusters
    print('Pickup clusters')
    cl_pu_avg = train_info.groupby('pickup_clusters').trip_duration.mean()
    cl_pu_avg = cl_pu_avg.reset_index()
    cl_pu_avg.columns = ['pickup_clusters', 'pickup_clusters_avg']
    df = pd.merge(left=df, right=cl_pu_avg, on='pickup_clusters', how='left')
    
    print('Dropoff clusters')
    cl_do_avg = train_info.groupby('dropoff_clusters').trip_duration.mean()
    cl_do_avg = cl_do_avg.reset_index()
    cl_do_avg.columns = ['dropoff_clusters', 'dropoff_clusters_avg']
    df = pd.merge(left=df, right=cl_do_avg, on='dropoff_clusters', how='left')
    
    ### Distances ###
    print('Add distances')
    # Manhattan rot
    df['distance_manhattan_rot'] = my_manhattan_distances(df.pickup_longitude_rot, 
                                                          df.dropoff_longitude_rot,
                                                          df.pickup_latitude_rot, 
                                                          df.dropoff_latitude_rot)
    # Manhattan pca
    df['distance_manhattan_pca'] = df['distance_pca0'] + df['distance_pca1']
    # Euclidean
    df['distance_euclidean'] = my_euclidean_distances(df.pickup_latitude, 
                                                      df.dropoff_latitude,
                                                      df.pickup_longitude, 
                                                      df.dropoff_longitude)
    
    # Fastest route
    df = pd.merge(left=df,
                  right=frame_fastest[['id', 'total_distance', 
                                       'total_travel_time', 'number_of_steps']],
                  on='id', how='left')
    
    # Same destination
    print('Add same destination marker')
    df['same_destination'] = (df.distance_euclidean == 0).astype(np.uint)
    
    # Remove outliers
    mask = (df.eval_set != 2) & (df.trip_duration > np.log(20*24*60*60))
    print('Delete outliers:', mask.astype(np.uint).sum())
    df = df[~mask]
    return df


# ## Train-validation split

# In[4]:


### Add features
frame_augm = add_features(frame, predict=False)

drop_cols = ['dropoff_datetime', 'eval_set', 'pickup_datetime']             + ['dropoff_pca1', 'month_avg', 'dropoff_longitude',                'dropoff_longitude_rot', 'pickup_longitude',                'pickup_longitude_rot', 'pickup_pca0',                'pickup_pca1', 'dropoff_pca0']
# Train
X_train = frame_augm[(frame_augm.eval_set==0)].copy()
train_id = X_train.pop('id')
y_train = X_train.pop('trip_duration')
X_train.drop(drop_cols, axis=1, inplace=True)
# Validation
X_valid = frame_augm[frame_augm.eval_set==1].copy()
valid_id = X_valid.pop('id')
y_valid = X_valid.pop('trip_duration')
X_valid.drop(drop_cols, axis=1, inplace=True)
print('Train shape:', X_train.shape, '\nTest shape:', X_valid.shape)


# ## LightDBM model
# 
# ### Important:
# 
# Since kernels should work **under 1200 seconds** in the next cell some parameters of the model were changed. For better CV try, for example, **learning_rate = 0.3**. Some other parameters can be optimized too.

# In[ ]:


def lgb_rmsle_score(preds, dtrain):
    labels = np.exp(dtrain.get_label())
    preds = np.exp(preds.clip(min=0))
    return 'rmsle', np.sqrt(np.mean(np.square(np.log1p(preds)-np.log1p(labels)))), False

d_train = lgb.Dataset(X_train, y_train)

lgb_params = {
    'learning_rate': 1.0, # try 0.2
    'max_depth': 8,
    'num_leaves': 55, 
    'objective': 'regression',
    #'metric': {'rmse'},
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    #'bagging_freq': 5,
    'max_bin': 200}       # 1000
cv_result_lgb = lgb.cv(lgb_params,
                       d_train, 
                       num_boost_round=5000, 
                       nfold=3, 
                       feval=lgb_rmsle_score,
                       early_stopping_rounds=50, 
                       verbose_eval=100, 
                       show_stdv=True)
n_rounds = len(cv_result_lgb['rmsle-mean'])
print('num_boost_rounds_lgb=' + str(n_rounds))


# In[ ]:


def dummy_rmsle_score(preds, y):
    return np.sqrt(np.mean(np.square(np.log1p(np.exp(preds))-np.log1p(np.exp(y)))))

# Train a model
model_lgb = lgb.train(lgb_params, 
                      d_train, 
                      feval=lgb_rmsle_score, 
                      num_boost_round=n_rounds)
# Predict on train
y_train_pred = model_lgb.predict(X_train)
print('RMSLE on train = {}'.format(dummy_rmsle_score(y_train_pred, y_train)))
# Predict on validation
y_valid_pred = model_lgb.predict(X_valid)
print('RMSLE on valid = {}'.format(dummy_rmsle_score(y_valid_pred, y_valid)))


# In[ ]:


plt.figure(figsize=(10,5))
# CV scores
plt.subplot(1,2,1)
train_scores = np.array(cv_result_lgb['rmsle-mean'])
train_stds = np.array(cv_result_lgb['rmsle-stdv'])
plt.plot(train_scores, color='green')
plt.fill_between(range(len(cv_result_lgb['rmsle-mean'])), 
                 train_scores - train_stds, train_scores + train_stds, 
                 alpha=0.1, color='green')
plt.title('LightGMB CV-results')
#plt.ylim(0.34,0.40)
plt.subplot(1,2,2)
plt.scatter(y_valid, y_valid_pred, s=0.2, alpha=0.7)
plt.plot([0,12], [0,12], color='g', alpha=0.3)
plt.xlabel('True (log) validation set'); plt.xlim(0,12)
plt.ylabel('Pred. (log) validation set'); plt.ylim(0,12)
plt.show()


# ### Feature importance

# In[ ]:


feature_imp = pd.Series(dict(zip(X_train.columns, model_lgb.feature_importance())))                     .sort_values(ascending=False)
feature_imp


# ### Predict

# In[ ]:


# Prediction on the test dataset
# Clean training
del frame_augm; gc.collect()

def test_predict():
    ### Add features
    frame_augm = add_features(frame, True)
    drop_cols = ['dropoff_datetime', 'eval_set', 'pickup_datetime']                 + ['dropoff_pca1', 'month_avg', 'dropoff_longitude',                    'dropoff_longitude_rot', 'pickup_longitude',                    'pickup_longitude_rot', 'pickup_pca0',                    'pickup_pca1', 'dropoff_pca0']
    # Train
    X_train = frame_augm[(frame_augm.eval_set!=2)].copy()
    train_id = X_train.pop('id')
    y_train = X_train.pop('trip_duration')
    X_train.drop(drop_cols, axis=1, inplace=True)
    # Test
    X_test = frame_augm[frame_augm.eval_set==2].copy()
    test_id = X_test.pop('id')
    X_test.drop(drop_cols + ['trip_duration'], axis=1, inplace=True)
    print('Train shape:', X_train.shape, 'Test shape', X_test.shape)

    # Train a model
    d_train = lgb.Dataset(X_train, y_train)
    model_lgb = lgb.train(lgb_params, 
                          d_train, 
                          feval=lgb_rmsle_score, 
                          num_boost_round=n_rounds)
    # Predict on validation
    y_test_pred = model_lgb.predict(X_test)
    return y_test_pred

### Predict on TEST
#y_test_pred = test_predict()
### Submission
#subm = pd.DataFrame()
#subm['id'] = test_id
#subm['trip_duration'] = np.exp(y_test_pred)
#subm.to_csv('submission.csv', index=False)

