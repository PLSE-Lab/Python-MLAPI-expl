#!/usr/bin/env python
# coding: utf-8

# This kernel can be used by starters in Machine Learning for understanding basic Data Preparation and modelling steps.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, asin, sin, cos, sqrt

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# loading the dataframe
def load_dataframe(path):
    return pd.read_csv(path, nrows = 100000)
train_dataframe = load_dataframe('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv')


# In[ ]:


train_dataframe.head()


# In[ ]:


train_dataframe.shape


# In[ ]:


train_dataframe['key'].nunique


# Inferences:
# 1. I am getting memory error so I am using just nrows = 100000.
# 2. The 'key' is nothing but the timestamp, which have the possiblity of repeating, 2 different taxis can be hired at the same time, exactly at the same time. However the probablity is extremely low so we can assume that the alternate hypothesis works here.
# 3. The 'fare_amount' is the target feature to be predicted.
# 4. The pickup location and dropoff location can be used to calculate the distance between pickup and dropoff points .
# 5. The datetime feature can be binned into categories such as morning, afternoon, evening, night (Since there are many rows we can bin it into more variables to get good insights - Evenings of Thursdays, Fridays are peak hours. Mornings of Mondays are peak hours)

# **The DISTANCE maybe misleading considering the reality because there might be different routes for short distance, there might be traffic, there might be diversions etc. Distances from point A to point B, in reality, cannot be a straight line. We are gonna use Haversine distance here to calculate the distance between latitude-longitude A to latitude-longitude B**

# Looks we can make the best use of all the features here.

# In[ ]:


# Checking for missing values
train_dataframe.isnull().sum()


# In[ ]:


# dropping missing values
def drop_missing_values(dataframe):
    return dataframe.dropna()
train_dataframe = drop_missing_values(train_dataframe)


# In[ ]:


train_dataframe.isnull().sum()


# We can use Haversin formula to calculate the distance between a latitude and longitute on earth. The formula can be understood by clicking [here](https://en.wikipedia.org/wiki/Haversine_formula).

# In[ ]:


def haversine_distance(latitudeA, longitudeA, latitudeB, longitudeB):
    radius_of_earth = 6371.8 #in kilometers
    # converting everything into radians
    latitudeA, longitudeA, latitudeB, longitudeB = radians(latitudeA), radians(longitudeA), radians(latitudeB), radians(longitudeB)
    # finding the difference between the latitudes and longitudes
    latitude_difference = latitudeB - latitudeA
    longitude_difference = longitudeB - longitudeA
    # applyin the haversine formulas
    haversin_latitude = (1 - cos(latitude_difference))/2
    haversin_longitude = (1 - cos(longitude_difference))/2
    haversin_teta = haversin_latitude + (cos(longitudeA) * cos(longitudeB) * haversin_longitude)
    # finding the distance
    distance = 2 * radius_of_earth * asin(sqrt(haversin_teta))
    return distance
# haversine_distance(latitudeA, longitudeA, latitudeB, longitudeB)


# In[ ]:


def distance_feature(dataframe):
    dataframe['distance'] = haversine_distance(0, 0, 0, 0)
    for i in range(len(dataframe)):
        dataframe['distance'].loc[i] = haversine_distance(dataframe['pickup_latitude'].loc[i],
                                                          dataframe['pickup_longitude'].loc[i],
                                                          dataframe['dropoff_latitude'].loc[i],
                                                          dataframe['dropoff_longitude'].loc[i])
    dataframe = dataframe.drop(['pickup_latitude',
                                            'pickup_longitude',
                                            'dropoff_latitude',
                                            'dropoff_longitude',], axis = 1)
    return dataframe
train_dataframe = distance_feature(train_dataframe)
train_dataframe.head()


# Now the distance is calculated and now we can go ahead with bunning the data and time. 

# In[ ]:


train_dataframe['pickup_datetime'].dtype
train_dataframe['pickup_datetime'].head()


# In[ ]:


def time_taken(dataframe):
    # we will first convert into timestamp and then we will bin this
    dataframe['time_taken'] = pd.to_datetime(dataframe['pickup_datetime']).dt.hour
    dataframe = dataframe.drop(['pickup_datetime'], axis = 1)
    # Converting the time taken into binned values for calculations.
    dataframe['time_taken'] = pd.cut(dataframe['time_taken'],
                                       bins=np.array([-1, 3, 6, 9, 12, 15, 18, 21, 24]),
                                       labels=[0,1,2,3,4,5,6,7])
    return dataframe
train_dataframe = time_taken(train_dataframe)
train_dataframe.head()


# Now we will go ahead with the regression model

# In[ ]:


# splitting the data into train and testing set
from sklearn.model_selection import train_test_split
X = train_dataframe.drop(['fare_amount', 'key'], axis = 1)
y = train_dataframe['fare_amount']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 101)


# In[ ]:


# Data is not linear at all and there are just 3 input features and 1 target feature. 
# So I am using Random Forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500)


# In[ ]:


# fitting the model
model.fit(X_train, y_train)


# In[ ]:


# prediction some sample data for validating the model
predictions = model.predict(X_val)


# In[ ]:


# checking the performance of the model
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(predictions, y_val)))


# Now the model is trained and evaluated. Now performing the same in test set and submitting the file

# In[ ]:


test_dataframe = load_dataframe('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')
test_dataframe.head()


# In[ ]:


test_dataframe = drop_missing_values(test_dataframe)
test_dataframe.isnull().sum()


# In[ ]:


test_dataframe = distance_feature(test_dataframe)
test_dataframe.head()


# In[ ]:


test_dataframe = time_taken(test_dataframe)
test_dataframe.head()


# In[ ]:


# predicting the fare amount
X = test_dataframe.drop(['key'], axis = 1)
fare_amount = model.predict(X)
fare_amount[:10]


# In[ ]:


submission = pd.DataFrame({'key':test_dataframe['key'], 'fare_amount':fare_amount})
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index = False)


# Thanks for viewing my kernel and for your kind encouragement by upvoting. 
# 
# Suggestions and Discussions are encouraged. We all learn from our mistakes !! 
