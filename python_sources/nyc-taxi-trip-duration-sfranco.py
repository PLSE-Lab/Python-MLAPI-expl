#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[ ]:


INPUT_PATH = Path('..','input')
TRAIN_PATH = Path(INPUT_PATH, 'train.csv')
TEST_PATH = Path(INPUT_PATH, 'test.csv')

train = pd.read_csv(TRAIN_PATH, index_col='id', parse_dates=['pickup_datetime', 'dropoff_datetime'])
train.head()


# In[ ]:


test = pd.read_csv(TEST_PATH, index_col='id', parse_dates=['pickup_datetime'])
test.head()


# # EDA

# Date extraction

# In[ ]:


def extract_date_info(df, cols):
    for col in cols:
        df[col + '_month'] = df[col].dt.month
        df[col + '_week'] = df[col].dt.week
        df[col + '_dow'] = df[col].dt.dayofweek
        df[col + '_hour'] = df[col].dt.hour
        df[col + '_date'] = df[col].dt.date
    return df


# In[ ]:


train = extract_date_info(train, ['pickup_datetime', 'dropoff_datetime'])
test = extract_date_info(test, ['pickup_datetime'])
train.head()


# In[ ]:


#it seems that we have some very long trips that prevent us from looking at the ditribution let's plot all trip < 2h
fig, ax = plt.subplots(figsize=(18,8))
(train.loc[train['trip_duration'] < 2*3600, 'trip_duration'] /60).hist(bins= 100, ax=ax);
#seems like most of the traject are below 40 mins


# In[ ]:


# I'm now going to add the distance in my dataframe using the flight distance as the crow flies
# got the distance function from: https://janakiev.com/blog/gps-points-distance-python/

def haversine(lat1, lon1, lat2, lon2):
    R = 6372800  # Earth radius in meters  
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 +         math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))


# In[ ]:


train['distance'] = train.apply(lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,8))
ax.plot(train.groupby('pickup_datetime_date').count())
ax.plot(test.groupby('pickup_datetime_date').count());


# The 2 dataframes exactly overlaps on time so I will use this in my feature engineering and calulate a average speed per day

# In[ ]:


# next step : Speed

# train['speed'] = (train['distance'] / train['trip_duration'])*3,6 


# In[ ]:


train.head()


# # Feature selection + preprocess

# In[ ]:


X = train.drop(columns=['pickup_datetime', 'dropoff_datetime', 'trip_duration', 
                                 'pickup_datetime_date', 'dropoff_datetime_month',
                                 'dropoff_datetime_week','dropoff_datetime_dow',
                                 'dropoff_datetime_hour', 'dropoff_datetime_date'])


# In[ ]:


le = LabelEncoder()
le.fit(X['store_and_fwd_flag'])
X['store_and_fwd_flag'] = le.transform(X['store_and_fwd_flag'])

y = np.log1p(train['trip_duration'])


# In[ ]:


X.info()


# # Processing test.set

# In[ ]:


X_test = test.drop(columns=['pickup_datetime', 'pickup_datetime_date'])

X_test['store_and_fwd_flag'] = le.transform(X_test['store_and_fwd_flag'])

X_test['distance'] = test.apply(lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)
X_test.info()


# In[ ]:


rf = RandomForestRegressor(n_estimators=150, n_jobs=-1)
#Loss =  cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#np.mean(np.sqrt(-Loss))


# In[ ]:


rf.fit(X, y)
log_pred = rf.predict(X_test)
pred = np.expm1(log_pred)


# In[ ]:


SUBMIT_PATH = Path(INPUT_PATH, 'sample_submission.csv')
submit = pd.read_csv(SUBMIT_PATH)

arr_id = submit['id']
submission = pd.DataFrame({'id': arr_id, 'trip_duration': pred})
submission.head()


# In[ ]:


fi_dict = {
    'feats': X.columns,
    'feature_importance': rf.feature_importances_
}
fi = pd.DataFrame(fi_dict).set_index('feats').sort_values(
    'feature_importance', ascending=False)
fi.sort_values(
    'feature_importance', ascending=True).tail(10).plot.barh();


# In[ ]:


submission.to_csv("submission.csv", index=False)

