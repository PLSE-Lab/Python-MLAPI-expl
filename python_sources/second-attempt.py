#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows = 1000000)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train['fare_amount'].describe()


# In[ ]:


train = train.drop(train[train['fare_amount']<0].index, axis=0)
train.shape


# In[ ]:


train['fare_amount'].describe()


# In[ ]:


train['fare_amount'].sort_values(ascending=False)


# In[ ]:


train['passenger_count'].describe()


# In[ ]:


train[train['passenger_count']>6]


# In[ ]:


train = train.drop(train[train['passenger_count']==208].index, axis = 0)


# In[ ]:


train['passenger_count'].describe()


# In[ ]:


train['pickup_latitude'].describe()


# In[ ]:


train[train['pickup_latitude']<-90]


# In[ ]:


train[train['pickup_latitude']>90]


# In[ ]:


train = train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90])).index, axis=0)


# In[ ]:


train[train['dropoff_latitude']<-180]|train[train['dropoff_latitude']>180]


# In[ ]:


train['diff_lat'] = ( train['dropoff_latitude'] - train['pickup_latitude']).abs()
train['diff_long'] = (train['dropoff_longitude'] - train['pickup_longitude'] ).abs()


# In[ ]:


train.isnull().sum()


# In[ ]:


train = train.dropna(how = 'any', axis = 'rows')


# In[ ]:


plot = train.iloc[:2000].plot.scatter('diff_long', 'diff_lat')


# In[ ]:


train = train[(train.diff_long < 5.0) & (train.diff_lat < 5.0)]


# In[ ]:


def get_input_matrix(df):
    return np.column_stack((df.diff_long, df.diff_lat, np.ones(len(df))))

train_X = get_input_matrix(train)
train_y = np.array(train['fare_amount'])

print(train_X.shape)
print(train_y.shape)


# In[ ]:


(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)
print(w)


# In[ ]:


test['diff_lat'] = ( test['dropoff_latitude'] - test['pickup_latitude']).abs()
test['diff_long'] = (test['dropoff_longitude'] - test['pickup_longitude'] ).abs()


# In[ ]:


test_X = get_input_matrix(test)


# In[ ]:


test_y = np.matmul(test_X, w).round(decimals = 2)


# In[ ]:


submission = pd.DataFrame()
submission["key"] = test.key
submission["fare_amount"] = test_y
submission.to_csv('submission.csv', index = False)


# In[ ]:




