#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install geopy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install pathos')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install azure-storage')
get_ipython().system('pip install reverse_geocoder')
get_ipython().system('pip install -U scikit-learn')


# In[ ]:


import pandas as pd
from azure.storage.file import FileService
import os

account_name = ''
account_key = ''
file_service = FileService(account_name=account_name, account_key=account_key)
def get_train_data():
    file_service.get_file_to_path('data', None, 'train.csv', 'train.csv')
def get_test_data():
    file_service.get_file_to_path('data', None, 'test.csv', 'test.csv')

if not os.path.exists('train.csv'):
    get_train_data()
if not os.path.exists('test.csv'):
    get_test_data()
dataframe = pd.read_csv('train.csv', index_col=0)
test_dataframe = pd.read_csv('test.csv', index_col=0)
print dataframe.head()


# In[ ]:


from geopy.distance import great_circle
from dateutil.parser import parse
import time
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
import numpy as np

# def get_borough(lat, lng, retries=3):
#     if retries <= -1: return None
#     try:
#         geolocator = Nominatim()
#         location = geolocator.reverse('{}, {}'.format(lat, lng))
#         return location.address.split(', ')[2]
#     except Exception as e:
#         print 'Too many requests. Waiting for 2 mins for {}, {}'.format(lat, lng)
#         time.sleep(30)
#         return get_borough(lat, lng, retries-1)
        
def transform_df(df, great_circle=great_circle, parse=parse, np=np):
    yes_func = lambda x: 1 if x == 'Y' else 0
    df['distance'] = df.apply(lambda row : great_circle((row['pickup_latitude'], row['pickup_longitude']), 
                                                        (row['dropoff_latitude'], row['dropoff_longitude'])).miles, axis=1)
                              
    df['month'] = df.apply(lambda row: parse(row['pickup_datetime']).month, axis=1)
    df['day'] = df.apply(lambda row: parse(row['pickup_datetime']).weekday(), axis=1)
    df['pickup_hour'] = df.apply(lambda row: parse(row['pickup_datetime']).hour, axis=1)
    df['store_and_fwd_flag'] = df.apply(lambda row: yes_func(row['store_and_fwd_flag']), axis=1)
    return df

def parallelize_dataframe(df, func, num_cores, num_partitions):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.clear()
    return df

new_df = parallelize_dataframe(dataframe, transform_df, cpu_count(), cpu_count())
new_df['trip_duration'] = new_df.apply(lambda row: np.log1p(row['trip_duration']), axis=1)
print new_df.head()


# In[ ]:


import reverse_geocoder as rg
from sklearn import preprocessing

# def get_borough_rev_geocoder(lat, lng):
#     return rg.search((lat, lng))[0]['name']
# dataframe['pickup_borough'] = dataframe.apply(lambda row: 
#                                               get_borough_rev_geocoder(row['pickup_latitude'], row['pickup_longitude']), axis=1)
def get_coords(df, lat_key, long_key):
    lats = df[lat_key].values.tolist()
    longs = df[long_key].values.tolist()
    coords = zip(lats, longs)
    return coords
def encode_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le
def add_borough(df):
    pickup_coords = get_coords(df, 'pickup_latitude', 'pickup_longitude')
    dropoff_coords = get_coords(df, 'dropoff_latitude', 'dropoff_longitude')
    pickup_boroughs = np.array([d['admin2'] for d in rg.search(pickup_coords)])
    dropoff_boroughs = np.array([d['admin2'] for d in rg.search(dropoff_coords)])
    df['pickup_borough'] = encode_labels(pickup_boroughs).transform(pickup_boroughs)
    df['dropoff_borough'] = encode_labels(dropoff_boroughs).transform(dropoff_boroughs)
    return df
new_df = add_borough(new_df)
print new_df.head()


# In[ ]:


features = ['vendor_id', 'month', 'day', 'pickup_hour', 'store_and_fwd_flag', 'distance', 'pickup_borough', 'dropoff_borough']
def scale_df(df, test=False):
    if test:
        customized_df = df[features]
        x = customized_df.values #returns a numpy array
    else:
        customized_df = df[features + ['trip_duration']]
        x = customized_df.values[:,:-1] #returns a numpy array
        

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    if not test:
        trip_duration_col = np.array([customized_df['trip_duration'].values])
        x_scaled = np.concatenate((x_scaled, trip_duration_col.T), 1)
    scaled_df = pd.DataFrame(x_scaled, columns=customized_df.columns)
    return scaled_df
scaled_df = scale_df(new_df)
print scaled_df.head()


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def rlmse_func(predicted, actual):
    return np.sqrt(np.mean(np.square(np.log(predicted+1.0) - np.log(actual+1.0))))

rlmse = make_scorer(rlmse_func, greater_is_better=False)

# param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01, 0.3],
#               'max_depth': range(4, 7),
#               'subsample': [0.8, 1],
#               'n_estimators': [1000, 2000]
#               }

X_train, y_train = scaled_df[features].values, scaled_df[['trip_duration']].values
xgb_model = XGBRegressor(objective='reg:linear', max_depth=7, learning_rate=0.3, n_estimators=2000, nthread=-1)
print -1.0*cross_val_score(xgb_model, X_train, y_train.ravel(), scoring=rlmse, cv=10, verbose=8).mean()


# In[ ]:


test_df = parallelize_dataframe(test_dataframe, transform_df, cpu_count(), cpu_count())
new_test_df = scale_df(add_borough(test_df), test=True)
print new_test_df.head()


# In[ ]:


xgb_model = XGBRegressor(objective='reg:linear', max_depth=7, learning_rate=0.3, reg_lambda = 1.5, n_estimators=2000, nthread=-1)
xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(new_test_df.values)
print predictions


# In[ ]:


test_trip_duration = np.expm1(abs(predictions))
result_df = pd.DataFrame({'id': test_dataframe.index.values, 'trip_duration': test_trip_duration})
result_df.to_csv('answer.csv', index=False)


# In[ ]:




