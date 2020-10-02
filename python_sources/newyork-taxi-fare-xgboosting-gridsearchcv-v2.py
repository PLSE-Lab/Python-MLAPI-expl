#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#loading dataset
train_iop_path='/kaggle/input/new-york-city-taxi-fare-prediction/train.csv'
test_iop_path='/kaggle/input/new-york-city-taxi-fare-prediction/test.csv'
dataset_train=pd.read_csv(train_iop_path, nrows=1000000, index_col='key')
dataset_test=pd.read_csv(test_iop_path, nrows=1000000, index_col='key')


# In[ ]:


print("dataset_test old size", len(dataset_test))

dataset_test = dataset_test[dataset_test.dropoff_longitude != 0]
print("new size", len(dataset_test))
dataset_test.head()


# In[ ]:


print("old size", len(dataset_train))

dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]
print("new size", len(dataset_train))


# In[ ]:


dataset_train.isnull().sum()


# In[ ]:


dataset_train.dropna(axis=0, inplace=True)


# In[ ]:


dataset_train.isnull().sum()


# In[ ]:


dataset_train['pickup_datetime'][0]


# In[ ]:


# dataset_train = dataset_train.sample(n=1000)


# In[ ]:


import re
re.split(':|-| ',dataset_train['pickup_datetime'][0])


# In[ ]:


from geopy.distance import geodesic
plang, plat, dlang, dlat = -73.973320, 40.763805, -73.981430, 40.743835
geodesic((plang, plat), (dlang, dlat)).km


# In[ ]:


from math import radians, cos, sin, asin, sqrt

def haversine(row):
    lon1, lat1 =  row.dropoff_longitude, row.dropoff_latitude
    lon2, lat2 = row.pickup_longitude, row.pickup_latitude
    # convert decimal degrees to radians 
#     print(lon1, lat1, lon2, lat2)
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    dist = c * r
#     print(dist)
    return dist

def compute_distance(row):
#     print((row.dropoff_latitude, row.dropoff_longitude), 
#                                        (row.pickup_latitude, row.pickup_longitude))
    
    distance = geodesic((row.dropoff_latitude, row.dropoff_longitude), 
                                       (row.pickup_latitude, row.pickup_longitude)).km
#     print(distance)
    return distance

def prepare_data(df):
    data = df.copy()
    data['pickup_year'] = data['pickup_datetime'].map(lambda x: int(re.split(':|-| ', x)[0]))
    data['pickup_month'] = data['pickup_datetime'].map(lambda x: int(re.split(':|-| ', x)[1]))
    data['pickup_day'] = data['pickup_datetime'].map(lambda x: int(re.split(':|-| ', x)[2]))
    data['pickup_hour'] = data['pickup_datetime'].map(lambda x: int(re.split(':|-| ', x)[3]))
#     data['distance'] = ( (data['dropoff_longitude'] - data['pickup_longitude']) ** 2 + 
#                                (data['dropoff_latitude']- data['pickup_latitude']) ** 2 ) **.5 
    
    data['distance'] = data.apply(haversine, axis='columns')
    
    data.drop(['pickup_datetime'], axis=1, inplace=True)

    data.drop(['pickup_longitude'], axis=1, inplace=True)
    data.drop(['pickup_latitude'], axis=1, inplace=True)
    data.drop(['dropoff_longitude'], axis=1, inplace=True)
    data.drop(['dropoff_latitude'], axis=1, inplace=True)
    return data 


# In[ ]:


train_df = prepare_data(dataset_train)


# In[ ]:


train_df.head()


# In[ ]:


import seaborn as sns


# In[ ]:


plt.figure(figsize=(16,8))
sns.distplot(train_df.fare_amount)


# In[ ]:


plt.figure(figsize=(16,8))
sns.scatterplot(y=train_df.distance, x=train_df.fare_amount)


# In[ ]:


plt.figure(figsize=(16,8))
sns.scatterplot(y=train_df.pickup_month, x=train_df.fare_amount)


# In[ ]:


plt.figure(figsize=(16,8))
sns.scatterplot(y=train_df.pickup_day, x=train_df.fare_amount)


# In[ ]:


plt.figure(figsize=(16,8))
sns.scatterplot(y=train_df.pickup_hour, x=train_df.fare_amount)


# In[ ]:


test_df = prepare_data(dataset_test)
test_df.head()


# In[ ]:


train_df.to_csv('train_prepared.csv')


# In[ ]:


test_df.to_csv('test_prepared.csv')


# In[ ]:


n_esitmators = list(range(100, 1001, 100))
print('n_esitmators', n_esitmators)
learning_rates = [x / 100 for x in range(5, 101, 5)]
print('learning_rates', learning_rates)


# In[ ]:


parameters = [{'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 
                     'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
                    }]


# In[ ]:


y_train = train_df.fare_amount
X_train =train_df.drop('fare_amount',axis=1)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
gsearch = GridSearchCV(estimator=XGBRegressor(),
                       param_grid = parameters, 
                       scoring='neg_mean_absolute_error',
                       n_jobs=4,cv=3)

gsearch.fit(X_train, y_train)


# In[ ]:


gsearch.best_params_.get('n_estimators'), gsearch.best_params_.get('learning_rate')


# In[ ]:


final_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'), 
                           learning_rate=gsearch.best_params_.get('learning_rate'), 
                           n_jobs=4)


# In[ ]:


final_model.fit(X_train, y_train)


# In[ ]:


test_preds = final_model.predict(test_df)


# In[ ]:


output = pd.DataFrame({'key': test_df.index,
                      'fare_amount': test_preds})
output.to_csv('submission.csv', index=False)
print('done')

