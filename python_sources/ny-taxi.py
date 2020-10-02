#!/usr/bin/env python
# coding: utf-8

# # Taxi New York

# In[ ]:


import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from geopy import distance
import os
from sklearn.model_selection import cross_val_score
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ## Data loading

# In[ ]:


#train = pd.read_csv("./input/nyc-taxi-trip-duration/train.csv")
#test = pd.read_csv("./input/nyc-taxi-trip-duration/test.csv")
#sample = pd.read_csv("./input/nyc-taxi-trip-duration/sample_submission.csv")
import os
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#sample = pd.read_csv("../input/sample_submission.csv")


# ## data exploration

# In[ ]:


train.shape,test.shape


# In[ ]:


train.head(10)


# In[ ]:


test.head(10)


# In[ ]:


test.describe()


# In[ ]:


train.describe()


# In[ ]:


train.isna().sum()


# In[ ]:


test.isna().sum()


# In[ ]:


plt.subplots(figsize=(18,7))
plt.title("Outliers")
train.boxplot()


# ## data cleaning

# In[ ]:


train[['trip_duration']].boxplot()


# In[ ]:


train[['trip_duration']].boxplot(vert=False)


# In[ ]:


train[['pickup_longitude','dropoff_longitude']].boxplot()


# In[ ]:


train = train.loc[train['dropoff_longitude']> -75]
train = train.loc[train['pickup_longitude']> -75]
train[['pickup_longitude','dropoff_longitude']].boxplot()


# In[ ]:


train = train.loc[train['dropoff_longitude']< -73]
train = train.loc[train['pickup_longitude']< -73]
train[['pickup_longitude','dropoff_longitude']].boxplot()


# In[ ]:


train[['pickup_latitude','dropoff_latitude']].boxplot()


# In[ ]:


train = train.loc[train['dropoff_latitude']>40.5]
train = train.loc[train['pickup_latitude']>40.5]
train[['pickup_latitude','dropoff_latitude']].boxplot()


# In[ ]:


train = train.loc[train['dropoff_latitude']<41]
train = train.loc[train['pickup_latitude']<41]
train[['pickup_latitude','dropoff_latitude']].boxplot()


# ## Make more data

# ### Creat column for Day of the week, minute, hour, day and month

# In[ ]:


train['pickup_datetime']= pd.to_datetime(train.pickup_datetime, format='%Y-%m-%d %H:%M:%S')
train['day_of_the_date']=train.pickup_datetime.dt.dayofweek
train['month'] = train.pickup_datetime.dt.month
train['day'] = train.pickup_datetime.dt.day
train['hour'] = train.pickup_datetime.dt.hour
train['minute'] = train.pickup_datetime.dt.minute
train.head(5)


# In[ ]:


test['pickup_datetime']= pd.to_datetime(test.pickup_datetime, format='%Y-%m-%d %H:%M:%S')
test['day_of_the_date']=test.pickup_datetime.dt.dayofweek
test['month'] = test.pickup_datetime.dt.month
test['day'] = test.pickup_datetime.dt.day
test['hour'] = test.pickup_datetime.dt.hour
test['minute'] = test.pickup_datetime.dt.minute
test.head(5)


# ### Calculate the distance between pickup point and dropoff point

# In[ ]:


def distancer(row):
    coords_1 = (row['pickup_latitude'], row['pickup_longitude'])
    coords_2 = (row['dropoff_latitude'], row['dropoff_longitude'])
    return distance.distance(coords_1, coords_2).km

train['distance'] = train.apply(distancer, axis=1)
test['distance'] = test.apply(distancer, axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Log trip duration

# In[ ]:


train['trip_duration_log']=np.log(train['trip_duration'].values)
plt.hist(train['trip_duration_log'],bins=50)


# ## Features selection

# In[ ]:


train.columns,test.columns


# In[ ]:


input_columns=['day_of_the_date', 'month', 'day', 'hour','distance', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
y=train['trip_duration_log']
X=train[input_columns]
X_test=test[input_columns]


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2, random_state=42)


# In[ ]:


X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


# ## Entrainement

# In[ ]:


# n_estimators=19, min_samples_split=2, min_samples_leaf=4, max_features='auto', bootstrap=True, verbose=2
rfr = RandomForestRegressor(n_estimators=100,min_samples_leaf=3, min_samples_split=15, n_jobs=-1, max_features="auto")
rfr.fit(X_train, y_train)


# In[ ]:


cv_scores = cross_val_score(rfr, X_train, y_train, cv=5)
for i in range (len (cv_scores)):
    cv_scores[i]=np.sqrt(abs(cv_scores[i]))
print(np.mean(cv_scores))


# ## Submission

# In[ ]:


train_pred=rfr.predict(X_test)
train_pred


# In[ ]:


len(train_pred)


# In[ ]:


my_submission = pd.DataFrame({'id':test.id, 'trip_duration':np.exp(train_pred)})


# In[ ]:


my_submission.to_csv('sub.csv',index=False)

