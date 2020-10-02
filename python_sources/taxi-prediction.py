#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd 
from scipy import stats
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## DATA IMPORT
# 

# In[ ]:


train_df =  pd.read_csv('../input/nyc-taxi-trip-duration/train.csv', nrows = 1_000_000)
train_df.dtypes


# ## Data Exploration
# 

# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


train_df['trip_duration'].describe()


# In[ ]:


y = np.log1p(train_df['trip_duration'])
   


# In[ ]:


y.hist(bins=100, figsize=(14,3))
plt.xlabel('during')
plt.title('Histogram');


# ## Featuring Engenering
# 

# In[ ]:


from haversine import haversine
def calcul_distance(df):
   pickedup = (df['pickup_latitude'], df['pickup_longitude'])
   dropoff = (df['dropoff_latitude'], df['dropoff_longitude'])
   return haversine(pickedup, dropoff)


# In[ ]:





# In[ ]:


train_df['distance'] = train_df.apply(lambda x : calcul_distance(x), axis = 1)


# In[ ]:


train_df['passenger_count'].value_counts()
train_df['vendor_id'].value_counts()


# In[ ]:


train_df['vendor_id'] = train_df['vendor_id'].astype('category').cat.codes


# In[ ]:


##from datetime import datetime
##train_df_da = pd.to_datetime(train_df['pickup_datetime'])
##train_df['month'] = train_df_da.dt.month
##train_df['hour'] = train_df_da.dt.hour
##train_df['wday'] = train_df_da.dt.weekday


# In[ ]:


train_df['speed'] = train_df['distance']/train_df['trip_duration']*3.6


# In[ ]:


train_df.describe()


# In[ ]:


train_df.dtypes


# In[ ]:





# In[ ]:


train_df['passenger_count']


# In[ ]:


train_df.dtypes


# In[ ]:


train_new_1 = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv")
train_new_2 = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv")
train_test = pd.read_csv("../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv")
train_new = pd.concat([train_new_1, train_new_2], axis=0)


# In[ ]:


train_new_1.shape


# In[ ]:


train_new_2.shape


# In[ ]:


train_new.dtypes


# In[ ]:


train_all = train_df.merge(train_new, on='id', how='inner')


# In[ ]:


train_all.dtypes


# In[ ]:


from datetime import datetime
train_d = pd.to_datetime(train_all['pickup_datetime'])
train_all['month'] = train_d.dt.month.astype('category').cat.codes
train_all['hour'] = train_d.dt.hour.astype('category').cat.codes
train_all['wday'] = train_d.dt.weekday.astype('category').cat.codes


# In[ ]:


train_all.shape


# In[ ]:


train_all.head()


# In[ ]:


train_all.dtypes


# ## Model Predict

# 

# In[ ]:


SELECTED_COLUMNS = ['vendor_id', 'passenger_count', 'distance', 'pickup_latitude','pickup_longitude','dropoff_latitude', 'dropoff_longitude','hour','month','wday','total_distance','total_travel_time']
X = train_all[SELECTED_COLUMNS]
X.head(15)


# In[ ]:


X.shape[0]


# In[ ]:





# In[ ]:


y = np.log1p(train_all['trip_duration'])


# In[ ]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 100 decision trees
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)


# In[ ]:


# Train the model on training data
rf.fit(X, y);


# In[ ]:


test_df =  pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
test_all = test_df.merge(train_test, on='id', how='inner')
test_df.dtypes
test_df.head(10)


# In[ ]:


test_all.dtypes


# In[ ]:


test_df['vendor_id'] = test_df['vendor_id'].astype('category').cat.codes


# In[ ]:


test_df['distance'] = test_df.apply(lambda x : calcul_distance(x), axis = 1)


# In[ ]:


from datetime import datetime
test_df_da = pd.to_datetime(test_df['pickup_datetime'])
test_df['month'] = test_df_da.dt.month.astype('category').cat.codes
test_df['hour'] = test_df_da.dt.hour.astype('category').cat.codes
test_df['wday'] = test_df_da.dt.weekday.astype('category').cat.codes


# In[ ]:


test_all = test_df.merge(train_test, on='id', how='inner')


# In[ ]:


from datetime import datetime
test_d_da = pd.to_datetime(test_all['pickup_datetime'])
test_all['month'] = test_d_da.dt.month.astype('category').cat.codes
test_all['hour'] = test_d_da.dt.hour.astype('category').cat.codes
test_all['wday'] = test_d_da.dt.weekday.astype('category').cat.codes


# In[ ]:


test_all.dtypes


# In[ ]:


X_test = test_all[SELECTED_COLUMNS]
X_test.describe()


# In[ ]:



predictions = np.exp(rf.predict(X_test))-np.ones(len(X_test))
X_test.shape
pred = pd.DataFrame(predictions, index=test_df['id'])
pred.columns = ['trip_duration']
pred.to_csv("dat1.csv")

pd.read_csv('dat1.csv').head()


# In[ ]:


X_test.shape


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = -cross_val_score(rf, X, y, cv=2, scoring = 'neg_mean_squared_error' )


# In[ ]:


#math.sqrt(scores.mean())


# In[ ]:





# In[ ]:




