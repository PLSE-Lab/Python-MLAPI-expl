#!/usr/bin/env python
# coding: utf-8

# ## Module imports

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Data loading

# In[ ]:


TRAIN_PATH = os.path.join("..", "input", "train.csv")
TEST_PATH = os.path.join("..", "input", "test.csv")

df_train = pd.read_csv(TRAIN_PATH,index_col=0)
df_test = pd.read_csv(TEST_PATH, index_col=0)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ## 2. Data exploration

# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# **id** - a unique identifier for each trip
# 
# **vendor_id** - a code indicating the provider associated with the trip record
# 
# **pickup_datetime** - date and time when the meter was engaged
# 
# **dropoff_datetime** - date and time when the meter was disengaged
# 
# **passenger_count** - the number of passengers in the vehicle (driver entered value)
# 
# **pickup_longitude** - the longitude where the meter was engaged
# 
# **pickup_latitude** - the latitude where the meter was engaged
# 
# **dropoff_longitude** - the longitude where the meter was disengaged
# 
# **dropoff_latitude** - the latitude where the meter was disengaged
# 
# **store_and_fwd_flag** - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
# 
# **trip_duration** - duration of the trip in seconds
# 
# Disclaimer: The decision was made to not remove dropoff coordinates from the dataset order to provide an expanded set of variables to use in Kernels.

# In[ ]:


df_train["pickup_datetime"].head()


# ## 3. Data preprocessing

# ### 3.a Outliers

# In[ ]:


df_train["trip_duration"].hist(bins=100)


# There is a large majority of the values near 0 - 10000.
# 
# Let's have a look on this :

# In[ ]:


df_train[df_train.trip_duration < 10000].trip_duration.hist(bins=100)


# In[ ]:


# We check which partitions of the data to study
# I think we would only take data which correspond to 95% of the data ?

DURATION_MAX = 3500 # With this value we are studying 99% of the data

df_train[df_train.trip_duration < DURATION_MAX].trip_duration.hist(bins=100)

print("Count of values >", DURATION_MAX, ": ", df_train[df_train.trip_duration > DURATION_MAX].trip_duration.count())
print("Count of values <=", DURATION_MAX, ": ", df_train[df_train.trip_duration <= DURATION_MAX].trip_duration.count())
print("Lost data : ", df_train[df_train.trip_duration > DURATION_MAX].trip_duration.count() / df_train[df_train.trip_duration <= DURATION_MAX].trip_duration.count() * 100, "%")

df_train = df_train[df_train.trip_duration < DURATION_MAX]


# I did this because I wanted to select only representative data, but it will be done in a further version.

# ### 3.b Missing values

# In[ ]:


df_train.isna().sum()


# In[ ]:


df_train.duplicated().sum()


# In[ ]:


df_train[df_train.duplicated()]


# We can't consider these "duplicated values" as needed to be removed.

# ## 4 Features engineering

# ### 4.a Creation

# In[ ]:


df_train["pickup_hour"] = pd.to_datetime(df_train.pickup_datetime).dt.hour
df_train["pickup_day_of_week"] = pd.to_datetime(df_train.pickup_datetime).dt.dayofweek


# In[ ]:


df_train.pickup_hour.hist(bins=47)


# In[ ]:


mean_trip_duration_by_hour = []

for i in df_train.pickup_hour.unique():
    mean_trip_duration_by_hour.append(np.mean(df_train[df_train.pickup_hour == i].trip_duration))


# In[ ]:


fig, ax = plt.subplots(figsize=(22,8))
ax.scatter(x=df_train[:1000].pickup_hour, y=df_train[:1000].trip_duration)
ax.bar(df_train.pickup_hour.unique(), mean_trip_duration_by_hour)
plt.show()


# About day of week :

# In[ ]:


df_train.pickup_day_of_week.hist(bins=13)


# In[ ]:


mean_trip_duration_by_day = []

for i in df_train.pickup_day_of_week.unique():
    mean_trip_duration_by_day.append(np.mean(df_train[df_train.pickup_day_of_week == i].trip_duration))


# In[ ]:


fig, ax = plt.subplots(figsize=(22,8))
ax.scatter(x=df_train[:1000].pickup_day_of_week, y=df_train[:1000].trip_duration)
ax.bar(df_train.pickup_day_of_week.unique(), mean_trip_duration_by_day)
plt.show()


# It doesn't seems to be a relevent study, I won't use it for training for the moment...

# In[ ]:


df_test["pickup_hour"] = pd.to_datetime(df_test.pickup_datetime).dt.hour
df_test["pickup_day_of_week"] = pd.to_datetime(df_test.pickup_datetime).dt.dayofweek


# ### 4.b Features selection

# In[ ]:


# 1st test using only basic columns
SELECTION = ["pickup_longitude", "dropoff_longitude", "pickup_latitude", "dropoff_latitude", "pickup_hour"]
TARGET = "trip_duration"


# 

# In[ ]:


X_train = df_train[SELECTION]
y_train = df_train[TARGET]
X_test = df_test[SELECTION]


# ## 5. Model selection

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


# In[ ]:


# m1 = RandomForestRegressor(n_estimators=10)
# m1.fit(X_train, y_train)


# In[ ]:


m2 = RandomForestRegressor(n_estimators=15)
m2.fit(X_train, y_train)


# In[ ]:


# m3 = RandomForestRegressor(n_estimators=20)
# m3.fit(X_train, y_train)


# In[ ]:


m4 = RandomForestRegressor(n_estimators=15, min_samples_leaf=100, min_samples_split=150)
m4.fit(X_train, y_train)


# ## 6. Validation method

# Cross validation (NB : This might not be necessary because of the large number of rows we're working on)

# In[ ]:


cv_scores_1 = cross_val_score(m2, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')
cv_scores_1


# In[ ]:


# cv_scores_2 = cross_val_score(m4, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')
# cv_scores_2


# In[ ]:


# def rmse(test, pred):
#     return np.sqrt(mean_squared_error(test, pred))

def get_err(score):
    err_test = []
    for i in range(len(score)):
        err_test.append(np.sqrt(abs(score[i])))
    return err_test

print(np.mean(get_err(cv_scores_1)))
# print(np.mean(get_err(cv_scores_2)))


# Results :
#   
#   cross val 1 :  0.4097527612968065
#   
#   cross val 2 : 0.43580030140648873

# ## 7. Predictions

# In[ ]:


y_test_pred = m2.predict(X_test)
print(y_test_pred[:10])


# In[ ]:


d = { "id": df_test.index, "trip_duration": y_test_pred}
submission = pd.DataFrame(d)
submission.head()


# ## 8. Submission

# In[ ]:


submission.to_csv("submission.csv", index=0)

