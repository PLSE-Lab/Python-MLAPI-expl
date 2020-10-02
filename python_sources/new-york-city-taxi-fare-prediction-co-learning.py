#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_train=pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=20000)


# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# In[ ]:


df_train.tail()


# In[ ]:


#Basic Stats of the data set
df_train.describe()


# 1. Fare_amount is negative and it doesn't seem to be realistic
# 2. few longitude and lattitude entries are off

# In[ ]:


#drop the negative value
print("old size: %d" % len(df_train))
df_train = df_train[df_train.fare_amount >=0]
print("New size: %d" % len(df_train))


# In[ ]:


#check missing value
df_train.isnull().sum()/len(df_train)


# In[ ]:


#see the distribution of fae amount
df_train.fare_amount.hist(bins=100,figsize=(16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")


# * Looks like the distribution is highly skewed and the frequency above 60 is very less
# * we will plot below 60 and above 60 separately

# In[ ]:


#lest see the distribution of fae amount less then 60
df_train[df_train.fare_amount<60].fare_amount.hist(bins=50,figsize=(16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")


# There are few points between 40 and 60 dollars which has slightly high frequency and that could be airport trips

# * we can see here that there are total 40 trips which are above 60 dollars
# * some of them might be outliers or few of them mightbe long trip from/to airport we will see it in later section

# In[ ]:


df_train[df_train.fare_amount > 60].shape


# * we can see here that there are total 40 trips which are above 50 dollars
# * some of them might be outliers or few of them might be long distance trip from/to airport, we will see it in later sectionn

# In[ ]:


#lest see the distribution of fare amount more than 60
df_train[df_train.fare_amount>60].fare_amount.hist(bins=100,figsize=(16,8))
plt.xlabel("Fare Amount")
plt.ylabel("Frequency")


# let also check passanger count distribution

# In[ ]:


df_train[df_train.passenger_count<6].passenger_count.hist(bins=20,figsize=(16,8))
plt.xlabel("Passanger Count")
plt.ylabel("Frequency")


# * Most of the trips are taken by single passanger
# * we will try to see if there is any relation between passanger count and fare amout

# In[ ]:


df_train[df_train.passenger_count==0].shape


# * We have 75 such cases where passanger count is zero, there can be two possibility
# * Passanger count is incorrectly populated
# * Taxi was not carrying any passanger, may be taxi was used for goods
# * we will look into test data set and finalize whether we should drop these cases or not.

# In[ ]:


plt.figure(figsize= (16,8))
sns.boxplot(x = df_train[df_train.passenger_count< 6].passenger_count, y = df_train.fare_amount)


# * As we can see from the box plot median price of each passanger counts looks similar except one record, There are few outliers we wil treat in cleaning section
# * we will try to see if there is any relationship between passanger count and fare amount using correlation factor

# In[ ]:


df_train[df_train.passenger_count < 6][['fare_amount','passenger_count']].corr()


# There is very weak correlation (0.009) between fare amount and passanger count

# In[ ]:


df_test=pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
df_test.head()


# In[ ]:


df_test.shape


# In[ ]:


#check for missing value
df_test.isnull().sum()


# there are no missing value in test data set

# In[ ]:


df_test.describe()


# We will store the minimum and maximum of the longitude and latitude from test data set and filter the train data set for those data points

# In[ ]:


min(df_test.pickup_longitude.min(),df_test.dropoff_longitude.min()), max(df_test.pickup_longitude.max(),df_test.dropoff_longitude.max())


# In[ ]:


min(df_test.pickup_latitude.min(),df_test.dropoff_latitude.min()), max(df_test.pickup_latitude.max(),df_test.dropoff_latitude.max())


# In[ ]:


#this function will also be used with the test set below
def select_within_test_boundary(df,BB):
    return (df.pickup_longitude>=BB[0])&(df.pickup_longitude<=BB[1])&(df.pickup_latitude>=BB[2])&(df.pickup_latitude<=BB[3])&(df.dropoff_longitude>=BB[0])&(df.dropoff_longitude<=BB[1])&(df.dropoff_latitude>=BB[2])&(df.dropoff_latitude<=BB[3])


# In[ ]:


BB=(-74.5, -72.8, 40.5, 41.8)
print('Old size: %d' %len(df_train))
df_train=df_train[select_within_test_boundary(df_train,BB)]
print('New size: %d'%len(df_train))


# * Now we have sliced the train data records as per the coordinates of the test data

# Manual Feature Engineering
# * Adding distance metrics
# * few time based variables

# In[ ]:


def prepare_time_features(df):
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    df['hour_of_day'] = df.pickup_datetime.dt.hour
#     df['week'] = df.pickup_datetime.dt.week
    df['month'] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
#     df['day_of_year'] = df.pickup_datetime.dt.dayofyear
#     df['week_of_year'] = df.pickup_datetime.dt.weekofyear
    df["weekday"] = df.pickup_datetime.dt.weekday
#     df["quarter"] = df.pickup_datetime.dt.quarter
#     df["day_of_month"] = df.pickup_datetime.dt.day
    
    return df


# In[ ]:




