#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

## Use this code block for before and after data work
##print('Old size: %d' % len(df_train))
##print('New size: %d' % len(df_train))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

PATH = '../input/train.csv'

# Any results you write to the current directory are saved as output.


# Dont know how many rows exactly, so good practice to load how many rows are first in dataset.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Method 1, using file.readlines. Takes about 20 seconds.\nwith open(PATH) as file:\n    n_rows = len(file.readlines())\n\nprint (f'Exact number of rows: {n_rows}')")


# Dataset is huge, so limiting rows input

# In[ ]:


# read data in pandas dataframe
df_train =  pd.read_csv('../input/train.csv', nrows = 50_000, parse_dates=["pickup_datetime"])
df_test =  pd.read_csv('../input/test.csv')


# Data Types 

# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


# list first few rows (datapoints)
df_train.dtypes


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# Convert Dates and Time Stamp

# In[ ]:


df_train['pickup_datetime']=pd.to_datetime(df_train['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')

df_test['pickup_datetime']=pd.to_datetime(df_test['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


#train data
df_train['pickup_date']= df_train['pickup_datetime'].dt.date
df_train['pickup_day']=df_train['pickup_datetime'].apply(lambda x:x.day)
df_train['pickup_hour']=df_train['pickup_datetime'].apply(lambda x:x.hour)
df_train['pickup_day_of_week']=df_train['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
df_train['pickup_month']=df_train['pickup_datetime'].apply(lambda x:x.month)
df_train['pickup_year']=df_train['pickup_datetime'].apply(lambda x:x.year)
df_train.head()


# In[ ]:


#test data
df_test['pickup_date']= df_test['pickup_datetime'].dt.date
df_test['pickup_day']=df_test['pickup_datetime'].apply(lambda x:x.day)
df_test['pickup_hour']=df_test['pickup_datetime'].apply(lambda x:x.hour)
df_test['pickup_day_of_week']=df_test['pickup_datetime'].apply(lambda x:calendar.day_name[x.weekday()])
df_test['pickup_month']=df_test['pickup_datetime'].apply(lambda x:x.month)
df_test['pickup_year']=df_test['pickup_datetime'].apply(lambda x:x.year)
df_test.head()


# Find NULLS

# In[ ]:


print(df_train.isnull().sum())


# In[ ]:


print(df_test.isnull().sum())


# In[ ]:


print('Old size: %d' % len(df_train))
df_train = df_train.dropna(how = 'any', axis = 'rows')
df_test = df_test.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(df_train))


# Delete long distances
# 

# In[ ]:


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(df_train)


# In[ ]:


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df_2):
    df_2['abs_diff_longitude'] = (df_2.dropoff_longitude - df_2.pickup_longitude).abs()
    df_2['abs_diff_latitude'] = (df_2.dropoff_latitude - df_2.pickup_latitude).abs()

add_travel_vector_features(df_test)


# In[ ]:


plot = df_train.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')


# In[ ]:


plot = df_test.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')


# One degree is about 69 miles, so lets just make this the NY area only so we dont have to worry about outliers.

# In[ ]:


print('Old size: %d' % len(df_train))
df_train_2 = df_train[(df_train.abs_diff_longitude < 0.25) & (df_train.abs_diff_latitude < 0.25)]
df_test_2 = df_test[(df_test.abs_diff_longitude < 0.25) & (df_test.abs_diff_latitude < 0.25)]
print('New size: %d' % len(df_train))


# In[ ]:


df_test.head()


# In[ ]:


df_train.shape


# In[ ]:


df_test.shape


# Lets look at fare distribution

# In[ ]:


plt.figure(figsize=(12,5))
sns.kdeplot(df_train_2['fare_amount']).set_title("Distribution of Trip Fare")


# Lets delete fares that are less than 0!

# In[ ]:


df_train_2=df_train_2.loc[df_train_2['fare_amount']>1]
df_train_2=df_train_2.loc[df_train_2['fare_amount']<75]
df_train_2.shape

#other option
#df_train = df_train[df_train.fare_amount > 0.0]


# In[ ]:


df_train_2.head()


# In[ ]:


df_test_2.head()


# In[ ]:


df_test_2.shape


# Fare Dist Scatterplot

# In[ ]:


plot = df_train_2.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')


# In[ ]:


#calculate trip distance in miles
def distance(lat1, lat2, lon1,lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


# In[ ]:


df_train_2['trip_distance']=df_train_2.apply(lambda row:distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)
df_test_2['trip_distance']=df_test_2.apply(lambda row:distance(row['pickup_latitude'],row['dropoff_latitude'],row['pickup_longitude'],row['dropoff_longitude']),axis=1)


# In[ ]:


df_train_2.head()


# In[ ]:


df_test_2.head()


# In[ ]:


plot = df_train_2.iloc[:2000].plot.scatter('fare_amount', 'trip_distance')


# In[ ]:


ax = sns.scatterplot(x="fare_amount", y="trip_distance", data=df_train_2)


# There shouldnt be 0 distance and high fares, lets delete this data

# In[ ]:


df_train_3=df_train_2.loc[df_train_2['trip_distance']>0]
df_test_3=df_test_2.loc[df_test_2['trip_distance']>0]
ax = sns.scatterplot(x="fare_amount", y="trip_distance",hue="passenger_count", data=df_train_3)
df_train_3.describe()


# **Predictive Modelling Time!! **
# 

# In[ ]:


df_train_3.columns


# In[ ]:


df_test_3.columns


# In[ ]:


df_train_4 = df_train_3.drop(['key','pickup_datetime','pickup_day_of_week','pickup_date','pickup_year','abs_diff_longitude', 'abs_diff_latitude'], axis = 1)
df_test_4 = df_test_3.drop(['key','pickup_datetime','pickup_day_of_week','pickup_date','pickup_year','abs_diff_longitude', 'abs_diff_latitude'], axis = 1)


# In[ ]:


df_train_4.columns


# In[ ]:


df_test_4.columns


# In[ ]:


x_train = df_train_4.iloc[:,df_train_4.columns!='fare_amount']
y_train = df_train_4['fare_amount'].values
x_test = df_test_4


# In[ ]:


y_train.shape


# Validate we have the right columns

# In[ ]:


x_train.info()


# In[ ]:


x_test.info()


# Scoring time!

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10,random_state=0)
rf.fit(x_train, y_train)
rf_predict = rf.predict(x_test)


# In[ ]:


print(rf_predict)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission.info()
submission.head()


# In[ ]:


submission['fare_amount'] = rf_predict


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission_2.csv', index=False)
submission.head(20)


# 
