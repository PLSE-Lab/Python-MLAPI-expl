#!/usr/bin/env python
# coding: utf-8

# New York City Taxi Trip Duration
# 
# https://www.kaggle.com/c/nyc-taxi-trip-duration

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/nyc-taxi-trip-duration"))

# Any results you write to the current directory are saved as output.


# In[4]:


# Import libraries
import datetime
import math

import geopy.distance

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Assess data

# Let's first start by loading the training data and inspect it

# In[5]:


train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv',
                   parse_dates=['pickup_datetime', 'dropoff_datetime'],
                   dtype={'store_and_fwd_flag':'category'})


# In[6]:


train.info()


# In[7]:


train.head()


# In[8]:


train.shape


# In[9]:


train.describe(include='all')


# In[10]:


test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',
                   parse_dates=['pickup_datetime'],
                   dtype={'store_and_fwd_flag':'category'})


# In[11]:


test.describe(include='all')


# ### Quality & Tidiness
# 
# #### Categorical column (store_and_fwd_flag)
# * Use one hot encoding for this column
# 
# #### Datetime columns (pickup_datetime)
# * datetime columns which are **pickup_datetime** should be split to 'month', 'week', 'dayofweek', 'hour' (We should not need to do this for and **dropoff_datetime** because they are likely to be the same and we already have the tripduration which we try to predict
# * Apply MinMaxScaler to all new columns
# 
# #### Location columns (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude)
# * To create a train data, create a new column **distance_km** to store a distance value in km computed from (pickup_longitude	pickup_latitude	dropoff_longitude dropoff_latitude) and drop those data that exceed .99 quantile in the train data to remove outlier. Note that we do not need to do this for the test data.
#   * This decision was made after exploring the train data
# * Find a distance between a pickup location and a dropoff location to **Time Square** and add new columns to capture this information
#   * This decision was made to improve to experiment whether it can improve an accuracy of the prediction
#   * Intuitively, trip duration for locations closer to a popular tourist location should have a longer trip duration because of a heavy traffic in the area
# * Apply StandardScaler for the distance
# 
# #### Passenger count
# * Apply MinMaxScaler to the **passenger_count**
# 
# #### Drop unused columns
# * **id** column can be dropped because we do not need it in training
# * **pickup_datetime** and **dropoff_datetime** must be dropped after all above are doneto train a model
# * Original columns that are scaled in the previous steps will be dropped since we won't use it 

# ## Data Wrangling
# 
# Transform the original data to follow our requirements in the Quality & Tidiness section above

# In[12]:


def create_datetime_columns(df, col_name, scaler=None):
    '''
    Create addtional datetime columns and scale them
    '''

    raw_data = pd.concat([df[col_name].dt.dayofweek,
                      df[col_name].dt.month,
                      df[col_name].dt.week,
                      df[col_name].dt.hour], axis=1)
    
    if not scaler:
        scaler = MinMaxScaler()
        scaler.fit(raw_data)
    
    scaled_data = scaler.transform(raw_data)

    df[col_name+ '_' + 'dayofweek'] = scaled_data[:, 0]
    df[col_name+ '_' + 'month'] = scaled_data[:, 1]
    df[col_name+ '_' + 'week'] = scaled_data[:, 2]
    df[col_name+ '_' + 'hour'] = scaled_data[:, 3]
        
    return df, scaler


# In[13]:


def get_distance_km(row):
    '''
    Get a distance in kilometers between a pickup and dropoff locations of a given rows
    '''
    coords_1 = (row.pickup_latitude, row.pickup_longitude)
    coords_2 = (row.dropoff_latitude, row.dropoff_longitude)
    
    return geopy.distance.geodesic(coords_1, coords_2).km


# In[14]:


def get_distance_pickup_to_timesquare_km(row):
    
    coords_timesquare = (40.7590, -73.9845)
    
    coords_pickup = (row.pickup_latitude, row.pickup_longitude)
    
    return geopy.distance.geodesic(coords_pickup, coords_timesquare).km


# In[15]:


def get_distance_dropoff_to_timesquare_km(row):
    
    coords_timesquare = (40.7590, -73.9845)
    
    coords_dropoff = (row.dropoff_latitude, row.dropoff_longitude)
    
    return geopy.distance.geodesic(coords_dropoff, coords_timesquare).km


# In[16]:


def transform_data(df, scalers=None, cleanData=False):
    
    if scalers:
        scaler_datetime = scalers['datetime']
        scaler_distance = scalers['distance']
        scaler_passenger_count = scalers['passenger_count']
    else:
        scaler_datetime = None
        scaler_distance = None
        scaler_passenger_count = None
    
    data_clean = df.copy()
    
    #### Categorical column (store_and_fwd_flag)
    # This column must be converted to a numerical value by 
    # using cat.codes and cast it to int
    data_clean = pd.concat([data_clean.drop('vendor_id', axis=1), 
                        pd.get_dummies(data_clean['vendor_id'], prefix='vendor_id')], axis=1)
    
    data_clean = pd.concat([data_clean.drop('store_and_fwd_flag', axis=1), 
                        pd.get_dummies(data_clean['store_and_fwd_flag'], prefix='store_and_fwd_flag')], axis=1)
    
    
    #### Datetime columns (pickup_datetime)
    # datetime columns which is **pickup_datetime**
    # should be split to 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'hour'
#     data_clean = create_datetime_columns(data_clean, 
#                                          ['pickup_datetime', 'dropoff_datetime'])
    # Only do get additional column for pickup_datetime should be enought because
    # They are typically on the same day
    data_clean, scaler_datetime = create_datetime_columns(data_clean, 'pickup_datetime', scaler=scaler_datetime)

    #### Location columns (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude)
    # Create a new column **distance_km** to store a distance value in km computed from (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude)
    data_clean['distance_km'] = data_clean.apply(lambda row: get_distance_km(row), axis=1)
    data_clean['dist_pickup_to_timesquare'] = data_clean.apply(lambda row: get_distance_pickup_to_timesquare_km(row), axis=1)
    data_clean['dist_dropoff_to_timesquare'] = data_clean.apply(lambda row: get_distance_dropoff_to_timesquare_km(row), axis=1)

    raw_distance = pd.concat([data_clean['distance_km'],
                          data_clean['dist_pickup_to_timesquare'], 
                          data_clean['dist_dropoff_to_timesquare']], axis=1)
    
    if not scaler_distance:
        scaler_distance = StandardScaler()
        scaler_distance.fit(raw_distance)
    
    scaled_distance = scaler_distance.transform(raw_distance)
    
    data_clean['scaled_distance_km'] = scaled_distance[:, 0]
    data_clean['scaled_dist_pickup_to_timesquare'] = scaled_distance[:, 1]
    data_clean['scaled_dist_dropoff_to_timesquare'] = scaled_distance[:, 2]
    
    if cleanData:
        # After doing the exploratory analysis, I found that there are outliers in the dataset
        # (there are trips that have 1k km) that could potentially cause an unexpected behavior
        # Hence, remove those outlier data before proceeding         
        data_clean = data_clean[data_clean.distance_km < data_clean.distance_km.quantile(0.99)]
    
    
    #### Passenger count
    # Apply MinMaxScaler to the **passenger_count**
    data_passenger_count = np.array(data_clean['passenger_count']).reshape(-1, 1)
    
    if not scaler_passenger_count:
        scaler_passenger_count = MinMaxScaler()
        scaler_passenger_count.fit(data_passenger_count)

    scaled_passenger_count = scaler_passenger_count.transform(data_passenger_count)
    data_clean['scaled_passenger_count'] = scaled_passenger_count[:,0]
    
    #### Drop unused columns
    # **id** column can be dropped because we do not need it in training
    # **pickup_datetime** and **dropoff_datetime** must be dropped after all above are done

    data_clean = data_clean.drop(['id', 
                                  'pickup_datetime',
                                  'distance_km',
                                  'dist_pickup_to_timesquare',
                                  'dist_dropoff_to_timesquare',
                                  'passenger_count'
                                 ], axis=1)
    
    # Test data does not have dropof_datetime column. Hence, skip it
    if data_clean.columns.contains('dropoff_datetime'):
        data_clean = data_clean.drop(['dropoff_datetime'], axis=1)
        
    
    out_scalers = {'datetime': scaler_datetime, 
                   'distance': scaler_distance,
                   'passenger_count': scaler_passenger_count}
    
    return data_clean, out_scalers


# In[17]:


# # Clean the train data
# # Comment it out and use the saved clean data instead if it is already created

# print('[{}] Start'.format(datetime.datetime.now()))

# %time data_clean, out_scalers = transform_data(train, cleanData=True)

# data_clean.reset_index().to_feather('data_clean')


# In[18]:


# # We will use a saved clean data from the previous session here
data_clean = pd.read_feather('../input/nyctaxi-clean-train-data/data_clean')


# In[19]:


data_clean.head()


# In[20]:


# Inspect the output dataframe
data_clean.sample(20)


# In[21]:


data_clean.info()


# ## Exploratory Data Analysis (EDA)

# Before starting the modeling process, let's explore the data a little more to better understand it

# ### Correlation

# In[22]:


corr = data_clean.corr()


# In[23]:


plt.figure(figsize=(8,6))

sns.heatmap(corr);


# In[24]:


corr.style.background_gradient(cmap='coolwarm')


# In[25]:


# Get all column names
data_clean.columns.tolist()


# ## Use Default RandomForestRegressor Model

# In[26]:


X = data_clean.drop(['trip_duration'], axis=1)
y = data_clean['trip_duration']


# In[27]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)


# In[28]:


X_train.shape, X_valid.shape


# In[29]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m, X_train, X_valid, y_train, y_valid):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor()

print('[{}] Start'.format(datetime.datetime.now()))

get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')


# In[ ]:


print('[{}] Start'.format(datetime.datetime.now()))


get_ipython().run_line_magic('time', 'print_score(m, X_train, X_valid, y_train, y_valid)')


# Let's try to predict **trip_duration** of the **train****** data

# In[ ]:


y_pred_train = m.predict(X_train)


# Let's try to predict **trip_duration** of the **validation** data

# In[ ]:


y_pred = m.predict(X_valid)


# Now, let's find of Root Mean Squared Logarithmic Error of the predicted data.
# This is an evaluation metrics defined in Kaggle
# 
# https://www.kaggle.com/c/nyc-taxi-trip-duration/overview/evaluation

# In[ ]:


# From https://stackoverflow.com/questions/46202223/root-mean-log-squared-error-issue-with-scitkit-learn-ensemble-gradientboostingre
def rmsle(y, y0):
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))


# Here is RMSLE of the train data

# In[ ]:


rmsle(y_train, y_pred_train)


# Here is RMSLE of the validation data

# In[ ]:


rmsle(y_valid, y_pred)


# ### Just for fun

# Let submit this simple model and submit it for the competition to see where we are in the leadership board!

# Inspect the test data first. Notice that the **dropoff_datetime** and **trip_duration** columns are not included in this dataset!

# In[ ]:


test.head()


# In[ ]:


test.shape


# We need to transform data before passing it to the model and this can be done by using the **transform_data** function

# In[ ]:


# # Clean the test data
# # Comment it out and use the saved clean data instead if it is already created

# print('[{}] Start'.format(datetime.datetime.now()))

# %time test_clean, _ = transform_data(test, scalers=out_scalers)

# test_clean.reset_index().to_feather('test_clean')


# Save the cleaned test data to a feather file so we can reuse it in the future

# In[ ]:


# # We will use a saved clean data from the previous session here
test_clean = pd.read_feather('../input/nyctaxi-clean-test-data/test_clean')


# Inspect data to ensure that we get all appropriate columns and no rows are removed

# In[ ]:


test_clean.head()


# In[ ]:


test_clean.info()


# In[ ]:


test_clean.shape


# Create features data

# In[ ]:


X_sub = test_clean.copy()


# Predict the trip duration

# In[ ]:


y_sub = m.predict(X_sub)


# Now, replace data in the **trip_duration** column of a dataframe created from the **sample_submission.csv** with our predicted data, and save it to a csv file for submission

# In[ ]:


df_sub = pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.csv')


# In[ ]:


df_sub.head()


# In[ ]:


df_sub['trip_duration'] = y_sub
df_sub.head()


# In[ ]:


df_sub.shape


# In[ ]:


df_sub.to_csv('submission_default_scaling.csv', index=False)

