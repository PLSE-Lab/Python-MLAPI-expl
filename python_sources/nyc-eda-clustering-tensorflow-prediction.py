#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


#installing Tensorflow for future use 
import tensorflow as tf
tf.reset_default_graph()


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


NYC_Taxi_train=pd.read_csv('../input/train.csv')
NYC_Taxi_test=pd.read_csv('../input/test.csv')


# In[ ]:


NYC_Taxi_train.head()


# In[ ]:


NYC_Taxi_train.columns


# In[ ]:


#Data fields

# id - a unique identifier for each trip
# vendor_id - a code indicating the provider associated with the trip record
# pickup_datetime - date and time when the meter was engaged
# dropoff_datetime - date and time when the meter was disengaged
# passenger_count - the number of passengers in the vehicle (driver entered value)
# pickup_longitude - the longitude where the meter was engaged
# pickup_latitude - the latitude where the meter was engaged
# dropoff_longitude - the longitude where the meter was disengaged
# dropoff_latitude - the latitude where the meter was disengaged
# store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
# trip_duration - duration of the trip in seconds


# In[ ]:


#univariate analysis (analysis of all features individually )


# In[ ]:


# Obervation with id
# checking duplicate with 'id' feature 
NYC_Taxi_train['id'].duplicated().value_counts()
NYC_Taxi_test['id'].duplicated().value_counts()
# unique id count in train:1458644
# unique id count in test: 625134


# In[ ]:


# Obervation with vendor id
NYC_Taxi_train['vendor_id'].duplicated().value_counts()
NYC_Taxi_test['vendor_id'].duplicated().value_counts()


# In[ ]:


NYC_Taxi_train.groupby('vendor_id')['vendor_id'].sum()
# two vendor's provide taxi as per data set in future we would like to explore are they providing taxi's in some perticular let-log(area of NYC)


# In[ ]:


# popularity of vendor 
NYC_Taxi_train.groupby('vendor_id')['vendor_id'].sum().plot(kind='bar',figsize=(8,6))


# In[ ]:


# Passenger_count trend 
NYC_Taxi_train['passenger_count'].value_counts().sort_values()


# In[ ]:


NYC_Taxi_train['passenger_count'].value_counts().sort_values().plot(kind='barh',figsize=(8,6))
# Observations:
# 1) 60 taxi running with out passenger :) 
# 2) mostly passenger travel alone or with one more passenger ,after that thrid largest count is of 5 passenger's in group


# In[ ]:


# analysis of trip duration 
NYC_Taxi_train['trip_duration'].isnull().value_counts()


# In[ ]:


NYC_Taxi_train['trip_duration'].max()


# In[ ]:


NYC_Taxi_train['trip_duration'].min()
# funny minimum trip_duration is 1 sec 


# In[ ]:


NYC_Taxi_train_alt=NYC_Taxi_train[NYC_Taxi_train['trip_duration']<120]
NYC_Taxi_train_alt['trip_duration'].count()
# total trips finished with in 2 mins = 27817


# In[ ]:


# create new columns trip duration in mins AND trip duration in hours
NYC_Taxi_train['trip_duration_in_min']=(NYC_Taxi_train['trip_duration']/60).round(1)
NYC_Taxi_train['trip_duration_in_hour']=(NYC_Taxi_train['trip_duration_in_min']/60).round(2)


# In[ ]:


NYC_Taxi_train['trip_duration_in_min'].mean()


# In[ ]:


NYC_Taxi_train['trip_duration_in_min'].min()


# In[ ]:


NYC_Taxi_train['trip_duration_in_min'].max()


# In[ ]:


# in train dataset some trip duration are very high (I consider them outliers and remove them before replotting it)
q = NYC_Taxi_train.trip_duration.quantile(0.99)
NYC_Taxi_train = NYC_Taxi_train[NYC_Taxi_train.trip_duration < q]
plt.figure(figsize=(8,6))
plt.scatter(range(NYC_Taxi_train.shape[0]), np.sort(NYC_Taxi_train.trip_duration.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('trip duration', fontsize=12)
plt.show()


# In[ ]:


# lets create a copy of NYC_Taxi_train with name "temp
temp=NYC_Taxi_train.copy()


# In[ ]:


temp = temp[temp.trip_duration < temp.trip_duration.quantile(0.995)] # Temporarily removing outliers
pickup_dates = pd.DatetimeIndex(temp['pickup_datetime'])


# In[ ]:


#ow are taxi rides split among day of week and hour of day?
weekday = pickup_dates.dayofweek
day, count = np.unique(weekday, return_counts = True)

plt.figure(figsize=(6,4))
ax = sns.barplot(x = day, y = count)
ax.set(xlabel = "Day of week", ylabel = "Count of taxi rides")
plt.show();


# In[ ]:


# lets check some trand analysis regarding specific month/specific day /perticular hour of day with regards to travel duration


# In[ ]:


def toDateTime( df ):
    
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    df['month'] = df['pickup_datetime'].dt.month
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_week'] = df['pickup_datetime'].dt.weekday_name
    
    return df


# In[ ]:


temp1=toDateTime(temp)


# In[ ]:


temp1.head()


# In[ ]:


temp1.columns


# In[ ]:


# lets play with distance among lat-long 


# In[ ]:


def locationFeatures( df ):
    #displacement
    df['y_dis'] = df['pickup_longitude'] - df['dropoff_longitude']
    df['x_dis'] = df['pickup_latitude'] - df['dropoff_latitude']
    
    #square distance
    df['dist_sq'] = (df['y_dis'] ** 2) + (df['x_dis'] ** 2)
    
    #distance
    df['dist_sqrt'] = df['dist_sq'] ** 0.5
    
    return df


# In[ ]:


train = locationFeatures(temp)
test = locationFeatures(temp)


# In[ ]:


train.head()


# In[ ]:


# So NOW ONWARDS our focus will be train file 

# LETS ENTER IN THE WORLD OF multivariate analysis


# In[ ]:


train.groupby(['day_week']).mean()[['month','trip_duration_in_min']].round(2)


# In[ ]:


train.groupby(['day_week']).mean()[['month','trip_duration_in_min']].round(2).plot(kind='barh')


# In[ ]:


df=train.pivot_table(index='day_week',columns='month',values='trip_duration_in_min',aggfunc='mean').round(2)


# In[ ]:


df


# In[ ]:


df.plot(kind='bar',figsize=(10,10))


# In[ ]:


train.groupby('hour').mean()['trip_duration_in_min'].round(0)


# In[ ]:


train.groupby('hour').mean()['trip_duration_in_min'].round(0).plot()


# In[ ]:


df1=train.pivot_table(index='day_week',columns='hour',values='trip_duration_in_min',aggfunc='mean').round(2)


# In[ ]:


df1


# In[ ]:


df1.plot(kind='barh',figsize=(10,10))


# In[ ]:


sns.heatmap(df1.corr())
plt.figure(figsize=(15,12))


# In[ ]:


# based on above graph we can easily figure it out that 8 to 6 are pick hours for NYC


# In[ ]:




