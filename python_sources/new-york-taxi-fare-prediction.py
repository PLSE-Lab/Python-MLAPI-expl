#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data= pd.read_csv('../input/train.csv',nrows=1000000,parse_dates=["pickup_datetime"])


# In[ ]:


train= data.copy(deep=True)
train.head()


# Key is not useful for us so we should drop it.

# In[ ]:


train.drop('key', axis=1, inplace=True)  


# In[ ]:


train.describe()


# 1. We see negative fare amounts which needs to be removed.
# 2. Passenger count of 208 doesn't make sense for a taxi.
# 3. If there are any null values then we should remove them as well.

# In[ ]:


#remove fare amounts less than 2
train= train[train['fare_amount']>2]


# In[ ]:


#check for null values
train.isnull().sum()


# In[ ]:


#remove nulls
train = train.dropna(how='any',axis=0) 


# In[ ]:


#lets check again
train.isnull().sum()


# In[ ]:


#lets see the passenger count
train.passenger_count.unique()


# We can see that 208 is an outlier so we should remove it.

# In[ ]:


train= train[train['passenger_count']<10]


# In[ ]:


#lets see the passenger count
train.passenger_count.unique()


# In[ ]:


train.describe()


# Lets see the distribution of fare amount.

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(train['fare_amount'])
plt.title('Fare distribution')


# 1. There are some outliers considering the longitude and latitude information.
# 2. It can be verified that New York City, NY, USA Latitude and longitude coordinates are: 40.730610, -73.935242. 
# https://www.latlong.net/place/new-york-city-ny-usa-1848.html
# 

# In[ ]:


train = train.loc[train['pickup_latitude'].between(40, 42)]
train = train.loc[train['pickup_longitude'].between(-75, -72)]
train = train.loc[train['dropoff_latitude'].between(40, 45)]
train = train.loc[train['dropoff_longitude'].between(-75, -72)]


# In[ ]:


#Initially we had 1 million rows
train.shape


# 1. It seems we have less no of features so we can generate some derived attributes to improve accuracy.
# 2. We can have the absolute difference between the latitude and longitude of pickup and dropoff locations.
# 3. We can also have a L2 and L1 distance between the pickup and dropoff locations as features.

# In[ ]:


train['latitude_diff']= (train['pickup_latitude']-train['dropoff_latitude']).abs()
train['longitude_diff']= (train['pickup_longitude']- train['dropoff_longitude']).abs()


# In[ ]:


train.describe()


# In[ ]:


#lets see how many of the rows have 0 absolute difference of latitude and longitude.
X=train[(train['latitude_diff'] == 0) & (train['longitude_diff'] == 0)]
X.shape


# In[ ]:


#lets add L2 and L1 distance as features in our data
train['L2']=  ((train['dropoff_latitude']-train['pickup_latitude'])**2 +
(train['dropoff_longitude']-train['pickup_longitude'])**2)**1/2

train['L1']= ((train['dropoff_latitude']-train['pickup_latitude']) +
(train['dropoff_longitude']-train['pickup_longitude'])).abs()


# In[ ]:


train.head()


# In[ ]:


# Lets see the correlation between features created
corr_mat = train.corr()
corr_mat.style.background_gradient(cmap='coolwarm')


# Here we can see latitude_diff, longitude_diff, L1 dist are correlated with fare_amount which we want to predict.

# In[ ]:


test_data= pd.read_csv('../input/test.csv',parse_dates=["pickup_datetime"])


# ## We need to create the same features for Test data as well.

# In[ ]:


test= test_data.copy(deep=True)
test.describe()


# It seems that the Test data doesn't have outliers and we just need to add the additional features.

# In[ ]:


test['latitude_diff']= (test['pickup_latitude']-test['dropoff_latitude']).abs()
test['longitude_diff']= (test['pickup_longitude']- test['dropoff_longitude']).abs()

test['L2']=  ((test['dropoff_latitude']-test['pickup_latitude'])**2 +
(test['dropoff_longitude']-test['pickup_longitude'])**2)**1/2

test['L1']= ((test['dropoff_latitude']-test['pickup_latitude']) +
(test['dropoff_longitude']-test['pickup_longitude'])).abs()


# In[ ]:


test.describe()
#train.head()


# # Training and Testing

# In[ ]:


train_x = train.drop(['pickup_datetime','fare_amount'],axis=1)
train_y = train['fare_amount'].values

test_x = test.drop(columns=['pickup_datetime','key'])


# In[ ]:


train_x.head()
#test_x.head()


# In[ ]:



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

LR = LinearRegression()

LR.fit(train_x, train_y)

#making predictions
lr_prediction= LR.predict(test_x)


submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = lr_prediction
submission.to_csv('submission_LR.csv', index=False)
submission.head(20)


# In[ ]:



from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(train_x, train_y)
RF_predict = RF.predict(test_x)

submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = RF_predict
submission.to_csv('submission_RF.csv', index=False)
submission.head(20)


# ## Lets test LightGBM

# In[ ]:


parameters = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': -1,
        'num_leaves': 25,
        'learning_rate': 0.02,
        'max_depth': -1,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'reg_aplha': 1,
        'reg_lambda': 0.001,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'verbose':0
    
    }


import lightgbm as lgbm

train_lgbm = lgbm.Dataset(train_x, train_y, silent=True)

lgbm_model= lgbm.train(parameters, train_lgbm, num_boost_round=500)

lgbm_prediction= lgbm_model.predict(test_x)

submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = lgbm_prediction
submission.to_csv('submission_LGBM.csv', index=False)
submission.head(20)


# Lets try XGboost method

# In[ ]:



import xgboost as xgb

xgb_train = xgb.DMatrix(train_x, label=train_y)
xgb_test = xgb.DMatrix(test_x)


params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.05
         }

xgb_model= xgb.train(params, xgb_train,50 )

xgb_prediction = xgb_model.predict(xgb_test)

submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = xgb_prediction
submission.to_csv('submission_XGB.csv', index=False)
submission.head(20)


# In[ ]:





# In[ ]:





# # Including time features
# 
# I have ignored time features initially and getting RMSE of around 3.7  We can do more feature engineering and add time features.

# In[ ]:


train_x['year'] =train['pickup_datetime'].dt.year
train_x['month'] = train['pickup_datetime'].dt.month
train_x['day']=train['pickup_datetime'].dt.day
train_x['day_of_week']=train['pickup_datetime'].dt.dayofweek
train_x['hour']=pd.to_datetime(train['pickup_datetime'], format='%H:%M').dt.hour


# In[ ]:


train_x.head()


# In[ ]:


# Doing same thing for test data
test_x['year'] =test['pickup_datetime'].dt.year
test_x['month'] = test['pickup_datetime'].dt.month
test_x['day']=test['pickup_datetime'].dt.day
test_x['day_of_week']=test['pickup_datetime'].dt.dayofweek
test_x['hour']=pd.to_datetime(test['pickup_datetime'], format='%H:%M').dt.hour


# In[ ]:


test_x.head()


# In[ ]:


# Lets see the correlation between features created
train_x['fare']= train['fare_amount']
corr_mat_new = train_x.corr()
corr_mat_new.style.background_gradient(cmap='coolwarm')


# In[ ]:


# We don't need this row anymore
train_x= train_x.drop(['fare'],axis=1)


# # We need to run all the models on this new data

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

LR = LinearRegression()

LR.fit(train_x, train_y)

#making predictions
lr_prediction= LR.predict(test_x)


submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = lr_prediction
submission.to_csv('submission_LR_new.csv', index=False)
submission.head(20)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor()
RF.fit(train_x, train_y)

RF_predict = RF.predict(test_x)

submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = RF_predict
submission.to_csv('submission_RF_new.csv', index=False)
submission.head(20)


# In[ ]:


parameters = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': -1,
        'num_leaves': 25,
        'learning_rate': 0.02,
        'max_depth': -1,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'reg_aplha': 1,
        'reg_lambda': 0.001,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'verbose':0
    
    }


import lightgbm as lgbm

train_lgbm = lgbm.Dataset(train_x, train_y, silent=True)

lgbm_model= lgbm.train(parameters, train_lgbm, num_boost_round=500)

#lgbm prediction
lgbm_prediction= lgbm_model.predict(test_x)

submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = lgbm_prediction
submission.to_csv('submission_LGBM_new.csv', index=False)
submission.head(20)


# In[ ]:


import xgboost as xgb

xgb_train = xgb.DMatrix(train_x, label=train_y)
xgb_test = xgb.DMatrix(test_x)


params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.05
         }

xgb_model= xgb.train(params, xgb_train,50 )

xgb_prediction = xgb_model.predict(xgb_test)

submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = xgb_prediction
submission.to_csv('submission_XGB_new.csv', index=False)
submission.head(20)

