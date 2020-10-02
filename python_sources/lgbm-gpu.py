#!/usr/bin/env python
# coding: utf-8

# Installing the gpu version of lgbm

# In[ ]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')
get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')
get_ipython().system('apt-get install -y -qq libboost-all-dev')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[ ]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')
get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# Installing the necessary libraries

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import lightgbm as lgbm

import gc
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Importing the dataset

# In[ ]:


dataset = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows = 25000000)
dataset = dataset.dropna(how = 'any', axis = 'rows')


# Removing the outliers

# In[ ]:


def clean_df(df):
    return df[(df.fare_amount > 2.5)  & (df.fare_amount <= 350) &
          (df.passenger_count > 0) & (df.passenger_count <= 6)  &
           ((df.pickup_longitude != 0) & (df.pickup_latitude != 0) & (df.dropoff_longitude != 0) & (df.dropoff_latitude != 0))]

dataset = clean_df(dataset)
dataset.describe()


# Based on the test data, we are removing some rows that are not useful for training since, the co-ordinates not present in the test data won't serve any purpose to the traning model.

# In[ ]:


#Training on range of latitude and longitude based on test data

BB = (-74.26, -72.98, 40.56, 41.70)

def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) &            (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) &            (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) &            (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])

dataset = dataset[select_within_boundingbox(dataset, BB)]


# Takes the pickup time column as a input and creates different columns for hour, day, month, year.

# In[ ]:


def add_datetime_info(dataset):
    # Convert to datetime format
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['year'] = dataset.pickup_datetime.dt.year
    return dataset


# The below ideas of adding airport distances and other important locations is taken from other notebooks.
# 
# Calculates the different types of distances between pickup and dropoff cordinates. Also calculating the distances from major locations like airport have proven to very effective in predicting the prices.

# In[ ]:


def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    # Define earth radius (km)
    R_earth = 6371
    
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                         [pickup_lat, pickup_lon, 
                                                          dropoff_lat, dropoff_lon])

    # Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    # Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))


def sphere_dist_bear(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    #Define earth radius (km)
    R_earth = 6371
    
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = pickup_lon - dropoff_lon
    
    #Compute bearing distance
    a = np.arctan2(np.sin(dlon * np.cos(dropoff_lat)),np.cos(pickup_lat) * np.sin(dropoff_lat) - np.sin(pickup_lat) * np.cos(dropoff_lat) * np.cos(dlon))
    return a


def add_airport_dist(dataset):
    """
    Return minumum distance from pickup or dropoff coordinates to each airport.
    JFK: John F. Kennedy International Airport
    EWR: Newark Liberty International Airport
    LGA: LaGuardia Airport
    SOL: Statue of Liberty 
    NYC: Newyork Central
    """
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892,-74.0445) # Statue of Liberty
    nyc_coord = (40.7141667,-74.0063889) 
    
    
    pickup_lat = dataset['pickup_latitude']
    dropoff_lat = dataset['dropoff_latitude']
    pickup_lon = dataset['pickup_longitude']
    dropoff_lon = dataset['dropoff_longitude']
    
    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)
    pickup_sol = sphere_dist(pickup_lat, pickup_lon, sol_coord[0], sol_coord[1]) 
    dropoff_sol = sphere_dist(sol_coord[0], sol_coord[1], dropoff_lat, dropoff_lon)
    pickup_nyc = sphere_dist(pickup_lat, pickup_lon, nyc_coord[0], nyc_coord[1]) 
    dropoff_nyc = sphere_dist(nyc_coord[0], nyc_coord[1], dropoff_lat, dropoff_lon)
    
    
    
    dataset['jfk_dist'] = pickup_jfk + dropoff_jfk
    dataset['ewr_dist'] = pickup_ewr + dropoff_ewr
    dataset['lga_dist'] = pickup_lga + dropoff_lga
    dataset['sol_dist'] = pickup_sol + dropoff_sol
    dataset['nyc_dist'] = pickup_nyc + dropoff_nyc
    
    return dataset

def radian_conv(degree):
    return  np.radians(degree)


# In[ ]:


dataset = add_datetime_info(dataset)
dataset = add_airport_dist(dataset)
                                  
dataset['distance'] = sphere_dist(dataset['pickup_latitude'], dataset['pickup_longitude'], 
                                   dataset['dropoff_latitude'] , dataset['dropoff_longitude']) 

dataset['bearing'] = sphere_dist_bear(dataset['pickup_latitude'], dataset['pickup_longitude'], 
                                   dataset['dropoff_latitude'] , dataset['dropoff_longitude'])

dataset['pickup_latitude'] = radian_conv(dataset['pickup_latitude'])
dataset['pickup_longitude'] = radian_conv(dataset['pickup_longitude'])
dataset['dropoff_latitude'] = radian_conv(dataset['dropoff_latitude'])
dataset['dropoff_longitude'] = radian_conv(dataset['dropoff_longitude'])


# In[ ]:


dataset.drop(["pickup_datetime", "key"], axis = 1, inplace = True)


# In[ ]:


y = dataset['fare_amount']
train = dataset.drop(columns=['fare_amount'])


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train, y, random_state = 123, test_size=0.10)


# In[ ]:


del dataset
del train
del y
import gc
gc.collect()


# In[ ]:


params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': 4,
        'num_leaves': 31,
        'learning_rate': 0.15,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 15,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,
        'num_rounds':50000,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
        
    }

train_set = lgbm.Dataset(x_train, y_train, silent=False, categorical_feature=['year','month','day'])
valid_set = lgbm.Dataset(x_test, y_test, silent=False, categorical_feature=['year','month','day'])
del x_train, y_train, x_test, y_test
gc.collect()
model = lgbm.train(params, train_set = train_set, num_boost_round=10000, early_stopping_rounds=500, verbose_eval=500, valid_sets=valid_set)


# In[ ]:


test_df = pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
test_df = add_datetime_info(test_df)
test_df = add_airport_dist(test_df)
test_df['distance'] = sphere_dist(test_df['pickup_latitude'], test_df['pickup_longitude'], 
                                   test_df['dropoff_latitude'] , test_df['dropoff_longitude'])

test_df['bearing_distance'] = sphere_dist_bear(test_df['pickup_latitude'], test_df['pickup_longitude'], 
                                   test_df['dropoff_latitude'] , test_df['dropoff_longitude'])

test_df['pickup_latitude'] = radian_conv(test_df['pickup_latitude'])
test_df['pickup_longitude'] = radian_conv(test_df['pickup_longitude'])
test_df['dropoff_latitude'] = radian_conv(test_df['dropoff_latitude'])
test_df['dropoff_longitude'] = radian_conv(test_df['dropoff_longitude'])
                                                                    
test_key = test_df['key']
test_df = test_df.drop(columns=['key', 'pickup_datetime'])

#Predict from test set
prediction = model.predict(test_df, num_iteration = model.best_iteration)      
submission = pd.DataFrame({
        "key": test_key,
        "fare_amount": prediction
})
submission.to_csv("submission.csv", index = False)


# In[ ]:


ax = lgbm.plot_importance(model, figsize=(10,10))
plt.show()

