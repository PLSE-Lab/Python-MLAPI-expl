#!/usr/bin/env python
# coding: utf-8

# # New York City Taxi Trip Duration

# The competition dataset is based on the 2016 NYC Yellow Cab trip record data made available in Big Query on Google Cloud Platform. The data was originally published by the NYC Taxi and Limousine Commission (TLC). The data was sampled and cleaned for the purposes of this playground competition. Based on individual trip attributes, participants should predict the duration of each trip in the test set.
# 
# ### Data fields
# 
# - `id` - a unique identifier for each trip
# 
# - `vendor_id` - a code indicating the provider associated with the trip record
# 
# - `pickup_datetime` - date and time when the meter was engaged
# 
# - `dropoff_datetime` - date and time when the meter was disengaged
# 
# - `passenger_count` - the number of passengers in the vehicle (driver entered value)
# 
# - `pickup_longitude` - the longitude where the meter was engaged
# 
# - `pickup_latitude` - the latitude where the meter was engaged
# 
# - `dropoff_longitude` - the longitude where the meter was disengaged
# 
# - `dropoff_latitude` - the latitude where the meter was disengaged
# 
# - `store_and_fwd_flag` - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
# 
# - `trip_duration` - duration of the trip in seconds

# ### Module imports

# In[ ]:


import os

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from datetime import datetime


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set({'figure.figsize':(10,6), 'axes.titlesize':20, 'axes.labelsize':8})


# ## 1 - Data loading

# In[ ]:


FILEPATH_TRAIN = os.path.join('../input/train.csv')
FILEPATH_TEST = os.path.join('../input/test.csv')


# In[ ]:


df_train = pd.read_csv(FILEPATH_TRAIN)
df_train.head()


# In[ ]:


df_test = pd.read_csv(FILEPATH_TEST)
df_test.head()


# ## 2 - Data exploration

# In[ ]:


df_train.info()


# In[ ]:


df_train.describe()


# ## 3 -  Data pre-processing

# ### 3.1 - Outliers

# In[ ]:


plt.hist(df_train[df_train.trip_duration < 5000].trip_duration, bins = 100)
plt.title('Trip duration distribution')
plt.xlabel('Duration of a trip (in seconds)')
plt.ylabel('Number of trips')
plt.show()


# All `trip_duration` are less than 5000 seconds, a majority of them are between 0 and 3000 seconds, that is why we decide to focus on these data : only trips of less than 3000 seconds

# In[ ]:


df_train = df_train[(df_train.trip_duration < 3000)]
df_train.info()


# We went from 1458644 entries against 1434783 after outliers management, a loss of less than 2% of the training data

# ### 3.2 - Missing & duplicate values

# In[ ]:


df_train.isna().sum()


# In[ ]:


df_train.duplicated().sum()


# There is no missing value or duplicated rows in the dataset

# ### 3.3 - Categorical variables

# The `store_and_fwd` column is a categorical variable, so we assign a numeric variable to each value in order to use it later
# 
# This change concerns training and test data

# In[ ]:


cat_vars = ['store_and_fwd_flag']


# In[ ]:


for col in cat_vars:
    df_train[col] = df_train[col].astype('category').cat.codes
df_train.head()


# In[ ]:


for col in cat_vars:
    df_test[col] = df_test[col].astype('category').cat.codes
df_test.head()


# ## 4 - Features engineering

# ### 4.1 - Features creation

# During the analysis of the outliers, it turns out that the distribution of the data is mostly on the right.
# 
# It seems relevant to be interested in the logarithmic value of the trip duration, hence the creation of the feature `log_trip_duration`

# In[ ]:


df_train['log_trip_duration'] = np.log(df_train.trip_duration)


# Moreover, it seems interesting to create a variable `distance` which represents the distance between the point of departure and arrival

# In[ ]:


df_train['distance'] = np.sqrt((df_train.pickup_latitude - df_train.dropoff_latitude)**2 + (df_train.pickup_longitude - df_train.dropoff_longitude)**2)


# In[ ]:


df_test['distance'] = np.sqrt((df_test.pickup_latitude - df_test.dropoff_latitude)**2 + (df_test.pickup_longitude - df_test.dropoff_longitude)**2)


# We are studying the logarithmic value of the duration trip, so we must do the same with the distance : `log_distance`

# In[ ]:


df_train['log_distance'] = np.log(df_train.distance)


# In[ ]:


df_test['log_distance'] = np.log(df_test.distance)


# ### 4.2 - Features selection

# In order to predict travel time, we are interested in some features
# 
# Therefor, we decide to ignore the following ones : `id`, `vendor_id`, and `store_and_fwd_flag`

# In[ ]:


df_train = df_train.drop(['vendor_id', 'store_and_fwd_flag'], axis=1)
df_train.head()


# In[ ]:


df_test = df_test.drop(['vendor_id', 'store_and_fwd_flag'], axis=1)
df_test.head()


# As mentioned above, we will only select the following features

# In[ ]:


num_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
target = 'log_trip_duration'


# In[ ]:


X_train = df_train.loc[:, num_features]
y_train = df_train[target]
X_test = df_test.loc[:, num_features]
X_train.shape, y_train.shape, X_test.shape


# ## 5 - Model training

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


m = RandomForestRegressor(n_estimators=20)
m.fit(X_train, y_train)


# ## 6 - Validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


cv_scores = cross_val_score(m,X_train,y_train,cv=5,scoring='neg_mean_squared_log_error')
cv_scores


# In[ ]:


for i in range(len(cv_scores)):
    cv_scores[i] = np.sqrt(abs(cv_scores[i]))
cv_scores


# ## 7 - Predictions

# In[ ]:


y_test_pred = m.predict(X_test)
y_test_pred[:5]


# ## 8 - Submit predictions

# In[ ]:


submission = pd.DataFrame({'id': df_test.id, 'trip_duration': np.exp(y_test_pred)})
submission.head()


# In[ ]:


submission.to_csv('Submission_file.csv', index=False)


# 

# 
