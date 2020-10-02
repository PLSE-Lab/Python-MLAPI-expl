#!/usr/bin/env python
# coding: utf-8

# # How much does a Taxi ride cost in NYC?

# I'm from Nepal and the fare rates are pretty much high in respect to the taxi fare/avg. person income ratio inside the country. I had a curosity of knowing how much do New Yorkers pay and will be paying (buhahaha) for their taxi fares.

# ## Importing the libraries as well as the data

# You can do some fancy stuffs to import the whole massive amount of data more systematically.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing the training set,

# In[ ]:


train_df = pd.read_csv('../input/train.csv', nrows = 100000)


# In[ ]:


train_df.shape


# The test data only has 9914 rows so it won't be taking much memory space.

# In[ ]:


test_df = pd.read_csv('../input/test.csv')


# In[ ]:


test_df.shape


# # Understanding the data

# Let me take a look at the head of the dataframe.

# In[ ]:


train_df.head(5)


# Checking for null values,

# In[ ]:


train_df.isnull().sum()


# Removing the null values as the numbers are too less to spend time trying to handle the missing values of data.

# In[ ]:


train_df.dropna(inplace=True)


# ## Anamoly detection

# Data this big certainly has to have some anomalies in it.

# In[ ]:


train_df.describe()


# The fare amount seems to be negative as well as the latitude and longitude are way too off. Let's quickly take care of it.

# In[ ]:


train_df = train_df[train_df['fare_amount']>0]


# Checking the shape again,

# In[ ]:


train_df.shape


# Let's get every data row with distance less than 15 miles out of the picture.

# In[ ]:


def distance(lat1, lon1, lat2, lon2):
    a = 0.5 - np.cos((lat2 - lat1) *  0.017453292519943295)/2 + np.cos(lat1 * 0.017453292519943295) * np.cos(lat2 * 0.017453292519943295) * (1 - np.cos((lon2 - lon1) *  0.017453292519943295)) / 2
    res = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return res


# In[ ]:


train_df['distance'] = distance(train_df.pickup_latitude, train_df.pickup_longitude,                                       train_df.dropoff_latitude,train_df.dropoff_longitude)


# Same for test set,

# In[ ]:


test_df['distance'] = distance(test_df.pickup_latitude, test_df.pickup_longitude,                                       test_df.dropoff_latitude,test_df.dropoff_longitude)


# In[ ]:


train_df = train_df[train_df['distance']<15]


# In[ ]:


train_df.describe()


# Remove data where there are no passengers or there are more than 10 passengers.

# In[ ]:


train_df = train_df[(train_df['passenger_count']!=0) & (train_df['passenger_count']<10)]


# Looks like we pretty much cleaned our training data! Time for some feature engineering.

# In[ ]:


# train_df['hour'] = train_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).hour)
# train_df['year'] = train_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).year)

# test_df['hour'] = test_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).hour)
# test_df['year'] = test_df.pickup_datetime.apply(lambda x: pd.to_datetime(x).year)


# Train test split,

# In[ ]:


feat_cols_s = ['distance','passenger_count']

X = train_df[feat_cols_s]
y = train_df['fare_amount']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# ## Modeling using Random Forest Regressor

# In[ ]:


# from sklearn.ensemble import RandomForestRegressor


# In[ ]:


# r_reg= RandomForestRegressor(n_estimators=500)


# In[ ]:


# r_reg.fit(X_train,y_train)


# In[ ]:


# y_pred_final = r_reg.predict(test_df[feat_cols_s])

# submission = pd.DataFrame(
#     {'key': test_df.key, 'fare_amount': y_pred_final},
#     columns = ['key', 'fare_amount'])
# submission.to_csv('Random Forest regression.csv', index = False)


# # Modeling using XGBoost

# In[ ]:


import xgboost as xgb


# In[ ]:


def XGBoost(X_train,X_test,y_train,y_test,num_rounds=500):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    return xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=dtrain,num_boost_round=num_rounds, 
                    early_stopping_rounds=20,evals=[(dtest,'test')],)


# In[ ]:


xgbm = XGBoost(X_train,X_test,y_train,y_test)
xgbm_pred = xgbm.predict(xgb.DMatrix(test_df[feat_cols_s]), ntree_limit = xgbm.best_ntree_limit)


# In[ ]:


submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount':xgbm_pred},
    columns = ['key', 'fare_amount'])
submission.to_csv('XGboost regression.csv', index = False)

