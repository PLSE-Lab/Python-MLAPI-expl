#!/usr/bin/env python
# coding: utf-8

# Import the library..

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
# import warnings
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from math import radians, cos, sin, asin, sqrt
from datetime import datetime


# Loading the data..

# In[ ]:


train= pd.read_csv('../input/train.csv', index_col=0)
test=pd.read_csv('../input/test.csv', index_col=0)
print("Total number of samples in train file : ", train.shape[0])
print("Total number of samples in test file : ", test.shape[0])


# In[ ]:


train.head()


# **Data Description**
# * id - a unique identifier for each trip
# * vendor_id - a code indicating the provider associated with the trip record
# * pickup_datetime - date and time when the meter was engaged
# * dropoff_datetime - date and time when the meter was disengaged
# * passenger_count - the number of passengers in the vehicle (driver entered value)
# * pickup_longitude - the longitude where the meter was engaged
# * pickup_latitude - the latitude where the meter was engaged
# * dropoff_longitude - the longitude where the meter was disengaged
# * dropoff_latitude - the latitude where the meter was disengaged
# * store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
# * trip_duration - duration of the trip in seconds, target variable

# **Train Data**

# In[ ]:


train.info()
train.shape


# In[ ]:


train.dtypes


# In[ ]:


train.describe()


# **Test Data**

# In[ ]:


test.head()


# In[ ]:


test.dtypes


# In[ ]:


test.shape


# **The data analysis**
# > > Data visualization

# In[ ]:


train.hist(bins=50, figsize=(20,15))
plt.show()


# Check the 'trip_duration' values

# In[ ]:


train.loc[train['trip_duration'] < 5000, 'trip_duration'].hist();

plt.title('trip_duration')
plt.show()


# In[ ]:


np.log1p(train['trip_duration']).hist();
plt.title('log_trip_duration')
plt.show()


# **Cleaning the data**

# In[ ]:


plt.subplots(figsize=(15,5))
train.boxplot()


# In[ ]:


train = train[(train.trip_duration < 5000)]


# In[ ]:


train.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude', alpha=0.1);


# In[ ]:


train = train.loc[(train['pickup_longitude'] > -75) & (train['pickup_longitude'] < -73)]
train = train.loc[(train['pickup_latitude'] > 40) & (train['pickup_latitude'] < 41)]


# In[ ]:


train.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude', alpha=0.1);


# In[ ]:


train = train.loc[(train['dropoff_longitude'] > -75) & (train['dropoff_longitude'] < -73)]
train = train.loc[(train['dropoff_latitude'] > 40.5) & (train['dropoff_latitude'] < 41.5)]


# In[ ]:


train['passenger_count'].hist(bins=100, log=True, figsize=(10,5));
plt.title('passenger_count')
plt.show()


# In[ ]:


train = train.loc[(train['passenger_count'] >= 0) & (train['passenger_count'] <= 6)]


# In[ ]:


train.isnull().sum()


# In[ ]:


train.duplicated().sum()


# In[ ]:


train = train.drop_duplicates()
train.duplicated().sum()


# In[ ]:


train.dtypes


# Convert str to datetime for "pickup_datetime" and "dropoff_datetime" 
# Drop "store_and_fwd_flag" 

# In[ ]:


train.drop(["store_and_fwd_flag"], axis=1, inplace=True)
test.drop(["store_and_fwd_flag"], axis=1, inplace=True)


# In[ ]:


train.shape, test.shape


# **Create the features**

# In[ ]:


plg, plt = 'pickup_longitude', 'pickup_latitude'
dlg, dlt = 'dropoff_longitude', 'dropoff_latitude'
pdt, ddt = 'pickup_datetime', 'dropoff_datetime'


# Calculate distance between two longitude-latitude.

# In[ ]:


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def euclidian_distance(x):
    x1, y1 = np.float64(x[plg]), np.float64(x[plt])
    x2, y2 = np.float64(x[dlg]), np.float64(x[dlt])    
    return haversine(x1, y1, x2, y2)


# In[ ]:


get_ipython().run_line_magic('time', '')
train['distance'] = train[[plg, plt, dlg, dlt]].apply(euclidian_distance, axis=1)


# In[ ]:


get_ipython().run_line_magic('time', '')
test['distance'] = test[[plg, plt, dlg, dlt]].apply(euclidian_distance, axis=1)


# In[ ]:


train[pdt] = train[pdt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
train[ddt] = train[ddt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))


# In[ ]:


test[pdt] = test[pdt].apply(lambda x : datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
#test dataset has not "dropoff_datetime"


# In[ ]:


train['month'] = train[pdt].apply(lambda x : x.month)
train['week_day'] = train[pdt].apply(lambda x : x.weekday())
train['day_month'] = train[pdt].apply(lambda x : x.day)
train['pickup_time_minutes'] = train[pdt].apply(lambda x : x.hour * 60.0 + x.minute)


# In[ ]:


test['month'] = test[pdt].apply(lambda x : x.month)
test['week_day'] = test[pdt].apply(lambda x : x.weekday())
test['day_month'] = test[pdt].apply(lambda x : x.day)
test['pickup_time_minutes'] = test[pdt].apply(lambda x : x.hour * 60.0 + x.minute)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape, test.shape


# In[ ]:


features_train = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance", "month", "week_day", "day_month", "pickup_time_minutes"]
X_train = train[features_train]
y_train = np.log1p(train["trip_duration"])

features_test = ["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance", "month", "week_day", "day_month", "pickup_time_minutes"]
X_test = test[features_test]


# XGBoost 

# In[ ]:


from xgboost.sklearn import XGBRegressor  
import scipy.stats as st
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestRegressor 


# In[ ]:


reg=XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=50, n_jobs=1)


# In[ ]:


reg.fit(X_train, y_train)


# In[ ]:


log_pred = reg.predict(X_test)
y_pred = np.exp(log_pred) - np.ones(len(log_pred))


# In[ ]:


my_submission = pd.DataFrame({'id': test.index, 'trip_duration': y_pred})
my_submission.head()


# In[ ]:


my_submission.to_csv("submission.csv", index=False)

