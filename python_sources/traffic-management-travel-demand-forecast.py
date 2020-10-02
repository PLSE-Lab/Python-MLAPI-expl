#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This notebook is a submission to **Grab AI For Sea Challenge - Traffic Management**, to forecast travel demand based on historical Grab bookings. 
# Challenge website: https://www.aiforsea.com/traffic-management
# 
# There are **four parts** in this notebook:
# * **Data cleaning & preprocessing**
# * **Model selection: Random Forest vs. XGBoost**
# * **Define a function to predict demands of T+1, ..., T+5 using known data till T**
# * **Predict demands of T+1, ..., T+5 using test data.** 
# 
# The test dataset can start from any time period after the timeframe of the training dataset. My model will use features from the test dataset ending at timestamp T and predict T+1 to T+5 for all the geohashes which appeared in the training dataset. 
# 
# Each time interval in this challenge is 15 minutes.
# 
# **For evaluators**: please uncomment the code in Part 4 and fill in the link of test dataset. The code will produce a CSV file containing the demand forecasts for T+1 to T+5 for all the geohashes from the training set. Please run all codes in this notebook to avoid any errors. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Part 1 - Data Cleaning & Preprocessing

# Take a look at training set:

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df_train = pd.read_csv('../input/training.csv')
df_train.head()


# Size of training data:

# In[ ]:


df_train.shape


# 1329 unique locations in the data

# In[ ]:


len(df_train.geohash6.unique())


# Convert timestamp into hours and mininutes:

# In[ ]:


df_train['hours'] = df_train['timestamp'].map(lambda x: int(x.split(':')[0]))
df_train['mins'] = df_train['timestamp'].map(lambda x: int(x.split(':')[1]))
df_train.head()


# Convert day, hours, mins into a single feature **"time"**:

# In[ ]:


df_train['time'] = 24*60*(df_train['day']-1) + 60*df_train['hours'] + df_train['mins']
df_train.head()


# Convert geohash6 into latitude and longtitude:

# In[ ]:


import Geohash
df_train['Latitude'] = df_train.geohash6.map(lambda x: float(Geohash.decode_exactly(x)[0]))
df_train['Longitude'] = df_train.geohash6.map(lambda x: float(Geohash.decode_exactly(x)[1]))
df_train = df_train.sort_values(by=['time','Latitude','Longitude'], ascending=True)
df_train = df_train.reset_index().drop('index',axis=1)
df_train.head()


# Not all locations appear in all time slots

# In[ ]:


df_train[['geohash6','demand']].groupby('geohash6').count().head(10)


# As the training set is a huge dataset with more than 4 million data, I will only use the last 14 days' data, out of which the last five timestamps are used for testing purpose and the rest is for training purpose.

# In[ ]:


max_day = df_train.day.max()
max_time = df_train.time.max()
train_start = df_train[df_train.day==61-13].index[0]
test_start = df_train[df_train.time==max_time-15*4].index[0]

Xtrain = df_train[['time', 'Latitude','Longitude']].iloc[train_start:test_start,:]
Xtest = df_train[['time', 'Latitude','Longitude']].iloc[test_start:,:]

ytrain = df_train.demand.iloc[train_start:test_start]
ytest = df_train.demand.iloc[test_start:]


# In[ ]:


Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape


# ## Part 2 - Model Selection

# ### Part 2.1 - RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor(n_estimators=30, max_depth=40)
model.fit(Xtrain, ytrain)
ytest_pred = model.predict(Xtest)
rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))
print('RMSE:',rmse)


# ### Part 2.2 - XGBRegressor

# In[ ]:


from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=35)
model.fit(Xtrain, ytrain)
ytest_pred = model.predict(Xtest)
rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))
print('RMSE:',rmse)


# #### From above output, XGBRegressor produces a smaller RMSE than RandomForestRegressor. Hence XGBRegressor will be used. 
# #### All the hyperparameters above have been refined.[](http://)

# Define a function to convert time into day, hour, minute and timestamp:

# In[ ]:


def convert_time(time):
    day = int(time/(24*60)) + 1
    hour = int((time-(day-1)*24*60)/60)
    minute = time-(day-1)*24*60-hour*60
    timestamp = ':'.join((str(hour),str(minute)))
    return (day, hour, minute, timestamp)


# ## Part 3 - Define a function to predict demands of T+1, ..., T+5 using known data till T 

# In[ ]:


def predict5ts(link, n_estimators=500, learning_rate=0.05, max_depth=35):
    df = pd.read_csv(link)
    df['hours'] = df['timestamp'].map(lambda x: int(x.split(':')[0]))
    df['mins'] = df['timestamp'].map(lambda x: int(x.split(':')[1]))
    df['time'] = 24*60*(df['day']-1) + 60*df['hours'] + df['mins']
    
    import Geohash
    df['Latitude'] = df.geohash6.map(lambda x: float(Geohash.decode_exactly(x)[0]))
    df['Longitude'] = df.geohash6.map(lambda x: float(Geohash.decode_exactly(x)[1]))

    df = df.sort_values(by=['time','Latitude','Longitude'], ascending=True)
    df = df.reset_index().drop('index',axis=1)
    
    X = df[['time', 'Latitude','Longitude']]
    y = df.demand
    
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X, y)
    
    T = df.time.max()
    T1 = T+15
    T2 = T+15*2
    T3 = T+15*3
    T4 = T+15*4
    T5 = T+15*5
    
    geohashes = df_train.geohash6.unique()
    geohashes2 = []
    latitudes = []
    longitudes = []
    times = []
    days = []
    timestamps = []

    for t in (T1,T2,T3,T4,T5):
        for gh in geohashes:
            geohashes2.append(gh)
            latitudes.append(float(Geohash.decode_exactly(gh)[0]))
            longitudes.append(float(Geohash.decode_exactly(gh)[1]))
            times.append(t)
            days.append(convert_time(t)[0])
            timestamps.append(convert_time(t)[-1])

    df_pred = pd.DataFrame({'geohash6': geohashes2, 'day': days, 'timestamp': timestamps,
                        'time': times, 'Latitude': latitudes, 'Longitude': longitudes})
    Xtest = df_pred[['time', 'Latitude','Longitude']]
    ypred = model.predict(Xtest)

    df_pred['demand'] = ypred
    output = df_pred[['geohash6', 'day', 'timestamp', 'demand']]
    output.to_csv('output.csv', index=False)


# Check if the above function works by testing a small portion of data from the training set.

# In[ ]:


df_trial = df_train[['geohash6','day','timestamp','demand']].iloc[-20000:,:]
df_trial.to_csv('df_trial.csv', index=False)

trial_link = 'df_trial.csv'
predict5ts(link=trial_link)

output = pd.read_csv('output.csv')
print(output.shape)
output.head()


# In[ ]:


os.remove("df_trial.csv")
os.remove("output.csv")


# ## Part 4 - Predict demands of T+1, ..., T+5 using test data
# * Please uncomment below code and enter the link of test data.
# * Below code will produce an output file **output.csv** which is the demand forecast of T+1,...,T+5 for all the geo-locations, where T is the last time stamp in the test data.

# In[ ]:


#test_link = '...'
#predict5ts(link=test_link)

