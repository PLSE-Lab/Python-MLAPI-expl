#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization library
import matplotlib.pyplot as plt # visualization library

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.


# ## Data Loading

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train.head(10)


# In[ ]:


train.tail(10)


# In[ ]:


train.info()


# In[ ]:


for x in train.keys():
    print(x)


# In[ ]:


train.isnull().sum()


# ## Data Exploration

# In a business point of view, we can firstly said :
# 
#     - the driver who extends the trip to earn more money (however, there is no taxi_fare column so no)
#     - the geographical position of the people between the beginning and the end of the race
#     - the time at which people took the taxi (e.g. during traffic jams or not; during the day or not etc.)
# 
# 

# In[ ]:


from datetime import datetime

train['pickup_datetime'] = train['pickup_datetime'].astype('datetime64[ns]')
train['dropoff_datetime'] = train['dropoff_datetime'].astype('datetime64[ns]')


# In[ ]:


pick_features = ['pickup_datetime', 'dropoff_datetime', 'vendor_id']
pick_df = train[pick_features].copy(True)
pick_df.head()


# In[ ]:


# Pull out the month, the week,day of week and hour of day and make a new feature for each

pick_df['week'] = pick_df.loc[:,'pickup_datetime'].dt.week;
pick_df['weekday'] = pick_df.loc[:,'pickup_datetime'].dt.weekday;
pick_df['hour'] = pick_df.loc[:,'pickup_datetime'].dt.hour;
pick_df['month'] = pick_df.loc[:,'pickup_datetime'].dt.month;

# Count number of pickups made per month and hour of day
month_usage = pd.value_counts(pick_df['month']).sort_index()
hour_usage = pd.value_counts(pick_df['hour']).sort_index()


# In[ ]:


figure = plt.subplot(2, 1, 2)
hour_usage.plot.bar(alpha = 0.5, color = 'orange')
plt.title('Pickups over Hour of Day', fontsize = 20)
plt.xlabel('hour', fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.xticks(rotation=0)
plt.yticks(fontsize = 18)
plt.show()


# In[ ]:


figure = plt.subplot(2, 1, 2)
month_usage.plot.bar(alpha = 0.5, color = 'pink')
plt.title('Pickups over Month', fontsize = 20)
plt.xlabel('Month', fontsize = 18)
plt.ylabel('Count', fontsize = 18)
plt.xticks(rotation=0)
plt.yticks(fontsize = 18)
plt.show()


# The pick hours of taxi trip are between 5 PM to 8 PM. During the night (from 12 AM to 7 AM) there is less taxi trip In terms of months, there are approximately as many users from January to June
# 

# In[ ]:


train.passenger_count.min()


# In[ ]:


train.passenger_count.max()


# There is 0 to 9 passengers by taxi trip. We will later drop the taxi trip with 0 passengers (because there must be atleast 1 passenger)
# 

# In[ ]:


train.plot.scatter(x='pickup_longitude',y='pickup_latitude')
plt.show()


# In[ ]:


train.plot.scatter(x='dropoff_longitude',y='dropoff_latitude')
plt.show()


# In[ ]:


train.trip_duration.min()


# In[ ]:


train.trip_duration.max()


# The trip duration's range is between 1 sec to 3526282 sec We will later adjust this range
# 

# ## Data preprocessing

# ### Outliers

# In[ ]:


train.boxplot(figsize=(15,10))
plt.show()


# In[ ]:


# As said before, there is no need to have the min (0 passenger), we will drop it
train = train[train['passenger_count']>= 1]


# In[ ]:


# The trip duration's range is between 1 sec to 3526282 sec
# We will drop values that are inferior to 1 min (60 sec) and superior to 166 min (10 000 sec).
train = train[train['trip_duration']>= 1 ]
train = train[train['trip_duration']<= 10000 ]


# In[ ]:


# We will drop the longitude and latitude (in pickup and dropoff that looks like outliers)
train = train.loc[train['pickup_longitude']> -90]
train = train.loc[train['pickup_latitude']< 47.5]

train = train.loc[train['dropoff_longitude']> -90]
train = train.loc[train['dropoff_latitude']> 34]


# ## Features engineering

# In[ ]:


col_diff = list(set(train.columns).difference(set(test.columns)))
col_diff


# In[ ]:


# To use the pickup and dropoff location, we will calculate the distance between them
train['dist'] = abs((train['pickup_latitude']-train['dropoff_latitude'])
                        + (train['pickup_longitude']-train['dropoff_longitude']))
test['dist'] = abs((test['pickup_latitude']-test['dropoff_latitude'])
                        + (test['pickup_longitude']-test['dropoff_longitude']))


# In[ ]:


y = train["trip_duration"]  # This is our target
X = train[["passenger_count","vendor_id", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "dist" ]]


# ## Model Selection

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


# In[ ]:


randf = RandomForestRegressor()


# In[ ]:


randf.fit(X, y)


# In[ ]:


shuffle = ShuffleSplit(n_splits=5, train_size=0.5, test_size=0.25, random_state=42)


# In[ ]:


cv_score = cross_val_score(randf, X, y, cv=shuffle, scoring='neg_mean_squared_log_error')
for i in range(len(cv_score)):
    cv_score[i] = np.sqrt(abs(cv_score[i]))
print(np.mean(cv_score))


# ## Prediction

# In[ ]:


test.head()


# In[ ]:


X_test = test[["vendor_id", "passenger_count","pickup_longitude", "pickup_latitude","dropoff_longitude","dropoff_latitude","dist"]]
prediction = randf.predict(X_test)
prediction


# In[ ]:


my_submission = pd.DataFrame({'id': test.id, 'trip_duration': prediction})
my_submission.head()


# In[ ]:


my_submission.to_csv('submission.csv', index=False)

