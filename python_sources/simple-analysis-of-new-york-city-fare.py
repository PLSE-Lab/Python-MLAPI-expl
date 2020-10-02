#!/usr/bin/env python
# coding: utf-8

# # New York City Taxi Fare Prediction
# ## Overview
# * train.csv - Input features and target fare_amount values for the training set (about 55M rows).
# * test.csv - Input features for the test set (about 10K rows). Your goal is to predict fare_amount for each row.
# * sample_submission.csv - a sample submission file in the correct format (columns key and fare_amount). This file 'predicts' fare_amount to be $11.35 for all rows, which is the mean fare_amount from the training set.

# ## ID
# * key - Unique string identifying each row in both the training and test sets. Comprised of pickup_datetime plus a unique integer, but this doesn't matter, it should just be used as a unique ID field. Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

# ## Features
# * **pickup_datetime** - timestamp value indicating when the taxi ride started.
# * **pickup_longitude** - float for longitude coordinate of where the taxi ride started.
# * **pickup_latitude** - float for latitude coordinate of where the taxi ride started.
# * **dropoff_longitude** - float for longitude coordinate of where the taxi ride ended.
# * **dropoff_latitude** - float for latitude coordinate of where the taxi ride ended.
# * **passenger_count** - integer indicating the number of passengers in the taxi ride.

# ## Target
# * **fare_amount** - float dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# ## Load Dataset

# In[ ]:


import pandas as pd
train = pd.read_csv('../input/train.csv', nrows=300_000)
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.shape


# In[ ]:


train.head()


# ## Preprocessing
# ### Datetime
# * Parsing Datetime

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt


# In[ ]:


train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['hour'] = train['pickup_datetime'].dt.hour
train['day'] = train['pickup_datetime'].dt.day
train['week'] = train['pickup_datetime'].dt.week
train['month'] = train['pickup_datetime'].dt.month
train['day_of_year'] = train['pickup_datetime'].dt.dayofyear
train['week_of_year'] = train['pickup_datetime'].dt.weekofyear


# In[ ]:


test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
test['hour'] = test['pickup_datetime'].dt.hour
test['day'] = test['pickup_datetime'].dt.day
test['week'] = test['pickup_datetime'].dt.week
test['month'] = test['pickup_datetime'].dt.month
test['day_of_year'] = test['pickup_datetime'].dt.dayofyear
test['week_of_year'] = test['pickup_datetime'].dt.weekofyear


# In[ ]:


train.head()
train = train.dropna(how = 'any', axis='rows')


# ### Longitude & Latitude
# * Calculate difference between longitude and latitude

# In[ ]:


train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 200)]
train = train.loc[(train['pickup_longitude'] > -75) & (train['pickup_longitude'] < 75)]
train = train.loc[(train['pickup_latitude'] > 40) & (train['pickup_latitude'] < 45)]
train = train.loc[(train['dropoff_longitude'] > -75) & (train['dropoff_longitude'] < 75)]
train = train.loc[(train['dropoff_latitude'] > 40) & (train['dropoff_latitude'] < 45)]
train = train.loc[train['passenger_count'] <= 8]


# In[ ]:


train['abs_diff_longitude'] = (train['pickup_longitude'] - train['dropoff_longitude']).abs()
train['abs_diff_latitude'] = (train['pickup_latitude'] - train['dropoff_latitude']).abs()


# In[ ]:


test['abs_diff_longitude'] = (test['pickup_longitude'] - test['dropoff_longitude']).abs()
test['abs_diff_latitude'] = (test['pickup_latitude'] - test['dropoff_latitude']).abs()


# ## Explore
# ### Datetime

# figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
# 
# figure.set_size_inches(18,8)
# 
# sns.barplot(data=train, x="hour", y="fare_amount", ax=ax1)
# sns.barplot(data=train, x="day", y="fare_amount", ax=ax2)
# sns.barplot(data=train, x="week", y="fare_amount", ax=ax3)
# sns.barplot(data=train, x="month", y="fare_amount", ax=ax4)
# sns.barplot(data=train, x="day_of_year", y="fare_amount", ax=ax5)
# sns.barplot(data=train, x="week_of_year", y="fare_amount", ax=ax6)

# In[ ]:


train.head()


# ### Difference between latitude and longitude

# In[ ]:


train.head()


# ### Passengercount

# In[ ]:


sns.barplot(data=train, x="passenger_count", y="fare_amount")


# ## Train

# In[ ]:


#'hour', 'passenger_count'
feature_names = ['hour', 'passenger_count','abs_diff_longitude', 'abs_diff_latitude']
feature_names


# In[ ]:


label_name = 'fare_amount'
label_name


# In[ ]:


X_train = train[feature_names]
y_train = train[label_name]
X_test = test[feature_names]


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
# Machine Learning
from sklearn.model_selection import train_test_split
import xgboost as xgb


# In[ ]:


#Linear Regression Model
regr = LinearRegression()
regr.fit(X_train, y_train)
regr_prediction = regr.predict(X_test)


# In[ ]:


#KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(X_train, y_train)
knr_prediction = knr.predict(X_test)


# In[ ]:


#Random Forest Model
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)


# In[ ]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)


# In[ ]:


#set parameters for xgboost
params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.05
         }
num_rounds = 50


# In[ ]:


xb = xgb.train(params, dtrain, num_rounds)


# In[ ]:


y_pred_xgb = xb.predict(dtest)
print(y_pred_xgb)


# In[ ]:


#Assigning weights
# predictions = (regr_prediction + rfr_prediction + knr_prediction + 3 * y_pred_xgb) / 6
predictions = y_pred_xgb


# In[ ]:


predictions


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = predictions


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('./simplenewyorktaxi.csv', index=False)

