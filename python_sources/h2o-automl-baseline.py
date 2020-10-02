#!/usr/bin/env python
# coding: utf-8

# ### 
# * forked from:  https://www.kaggle.com/i150077/automl-pred

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

from sklearn.metrics import mean_squared_error as msq
import h2o
from h2o.automl import H2OAutoML

print(os.listdir("../input"))


# In[ ]:


DEBUG = True

ROW_SAMPLE = 9123456

if DEBUG == True:
    ROW_SAMPLE =    1234 


# In[ ]:


train = pd.read_csv('../input/train.csv', nrows = ROW_SAMPLE)
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.drop('key',axis=1,inplace=True)
# train.pop('pickup_datetime')
gc.collect()


# In[ ]:


kID = test['key']
# test.pop('pickup_datetime')


# In[ ]:


# Clean dataset from https://www.kaggle.com/gunbl4d3/xgboost-ing-taxi-fares
# train = train[train['fare_amount']>0]
# train = train[train['fare_amount']<300]

def clean_df(df):
    return df[(df.fare_amount > 0) &  (df.fare_amount < 400) &
            (df.pickup_longitude > -80) & (df.pickup_longitude < -70) &
            (df.pickup_latitude > 35) & (df.pickup_latitude < 45) &
            (df.dropoff_longitude > -80) & (df.dropoff_longitude < -70) &
            (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45)]

train = clean_df(train)


# In[ ]:


train.shape


# In[ ]:


train.isnull().values.any()


# In[ ]:


train.dropna(inplace=True)
train.shape


# In[ ]:


test.isnull().values.any()


# In[ ]:


print(train.isnull().sum())


# In[ ]:


# y = train.pop('fare_amount') # pop also removes the column


# In[ ]:


train.head()


# In[ ]:


val = train.shape[0]
test.shape, train.shape


# In[ ]:


# train = pd.concat([train,test])
# pas = train['passenger_count']
# train.shape


# In[ ]:


# changed/simplified code
# the other variables calced here are also likley relevant. skip for now, as I want an exact comparison with the naive distance baseline

def get_dist(df):
    R = 6373.0

#     dlon = df['dropoff_longitude'] - df['pickup_longitude']
#     dlat =  df['dropoff_latitude'] - df['pickup_latitude']
    lat1 = train['pickup_latitude']
    lon1 = train['pickup_longitude']
    lat2 = train['dropoff_latitude']
    lon2 = train['dropoff_longitude']
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = ((np.sin(dlat/2.0))**2) + (np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2.0))**2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
#     del R,c,a,dlon,dlat
#     distance.shape
    return distance


# In[ ]:


train["distance"] = get_dist(train)
test["distance"] = get_dist(test)


# # Export our "small" data sample

# In[ ]:


# train.to_csv("train_9m_fares_dist.csv.gz",index=False,compression="gzip")
# test.to_csv("test_fares_dist.csv.gz",index=False,compression="gzip")


# ## import H2o and build a model on the numerics:
# 

# In[ ]:


h2o.init(max_mem_size = 9)
h2o.remove_all()


# In[ ]:


train.head()


# In[ ]:


tr = h2o.H2OFrame(train[["passenger_count","distance","fare_amount"]].values)


# In[ ]:


tr.head(3)


# In[ ]:


ts = h2o.H2OFrame(test[["passenger_count","distance"]].values)


# In[ ]:


ts.head(3)


# In[ ]:


model = H2OAutoML()


# In[ ]:


model.train(ts.col_names, 'C3', tr)


# In[ ]:


pred = model.predict(ts)


# In[ ]:


pred = pred.as_data_frame()


# In[ ]:


pred.head()


# In[ ]:


submission = pd.DataFrame({
        "key": kID,
#         "fare_amount": pred.values
    "fare_amount": pred.predict
})

submission.to_csv('FARES.csv',index=False)


# In[ ]:




