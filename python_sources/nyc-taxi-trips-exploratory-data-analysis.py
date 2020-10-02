#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis - NYC Taxt Trip Duration
# 
# 
# Kaggle Competation Link: https://www.kaggle.com/c/nyc-taxi-trip-duration/
# 
# 1. [Problem Statement](#1.-Problem-Statement)
# 2. [Python Libraries](#2.-Python-Libraries)
# 3. [Datasets](#3.-Datasets)
#     * 3.1. [Data Dictionary](#3.1.-Data-Dictionary)
#     * 3.2. [Dataset Overview](#3.2.-Dataset-Overview)
# 4. [Missing Values](#4.-Missing-Values)
# 5. [Open-Questions/Hypothesis](#5.-Open-Questions/Hypothesis)
#     * 5.1. [Difference in the columns between the training and test datasets](#5.1.-Difference-in-the-columns-between-the-training-and-test-datasets)
#     * 5.2. [Are all the id's in the train and test datasets unique? Is there an overlap, in the observations, between the train and test datasets?](#5.2.-Are-all-the-id's-in-the-train-and-test-datasets-unique?-Is-there-an-overlap,-in-the-observations,-between-the-train-and-test-datasets?)
#     * 5.3. [Are all the vendor_id's in the train and test datasets unique?](#5.3.-Are-all-the-vendor_id's-in-the-train-and-test-datasets-unique?)
#     * 5.4. [Distribution of the number of passengers across the vendor_id variables 1 and 2, in both the train and test datasets](#5.4.-Distribution-of-the-number-of-passengers-across-the-vendor_id-variables-1-and-2,-in-both-the-train-and-test-datasets)
#     * 5.5. [Distribution of the trip_duration across the train dataset](#5.5.-Distribution-of-the-trip_duration-across-the-train-dataset)
#     * 5.6. [Distribution of the store_and_fwd_flag across the train dataset](#5.6.-Distribution-of-the-store_and_fwd_flag-across-the-train-dataset)
#     * 5.7. [Exploring the distances between the pickup and dropoff lat/log coordinates](#5.7.-Exploring-the-distances-between-the-pickup-and-dropoff-lat/log-coordinates)
#     * 5.8. [Exploring the number of trips at each timestamp feature in the train dataset](#5.8.-Exploring-the-number-of-trips-at-each-timestamp-feature-in-the-train-dataset)
#     * 5.9. [Exploring the behaviour of trip_duration based on the number of trips for each timestamp feature in the train dataset](#5.9.-Exploring-the-behaviour-of-trip_duration-based-on-the-number-of-trips-for-each-timestamp-feature-in-the-train-dataset)
#     
# [References](#References)

# ## 1. Problem Statement
# 
# In this competition, the challenge is to build a model that predicts the total ride duration of taxi trips in New York City.
# 
# Feel free to provide Suggestions, Feedback and Upvotes :)

# ## 2. Python Libraries

# In[ ]:


# #Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels
import pandas_profiling

from sklearn import linear_model

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
import time
import requests
import datetime

import missingno as msno


# ## 3. Datasets

# In[ ]:


# #Datasets
get_ipython().system('ls ../input/')


# In[ ]:


# #Train and Test Datasets
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

df_sample_submission = pd.read_csv("../input/sample_submission.csv")


# ### 3.1. Data Dictionary
# 
# Dataset: input/train.csv
# 
# * id - a unique identifier for each trip
# * vendor_id - a code indicating the provider associated with the trip record
# * pickup_datetime - date and time when the meter was engaged
# * dropoff_datetime - date and time when the meter was disengaged
# * passenger_count - the number of passengers in the vehicle (driver entered value)
# * pickup_longitude - the longitude where the meter was engaged
# * pickup_latitude - the latitude where the meter was engaged
# * dropoff_longitude - the longitude where the meter was disengaged
# * dropoff_latitude - the latitude where the meter was disengaged
# * store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
# * trip_duration - duration of the trip in seconds
# 
# Dataset: input/test.csv
# 
# * The train dataset contains 11 columns and the test dataset contains 9 columns. The two additional columns that are present in the train dataset, and not in the test dataset are dropoff_datetime and trip_duration. 
# 
# Dataset: input/sample_submission.csv
# 
# * id - a unique identifier for each trip
# * trip_duration - duration of the trip in seconds

# ### 3.2. Dataset Overview

# #### Training Dataset

# In[ ]:


print("Total number of samples in train dataset: ", df_train.shape[0])
print("Number of columns in train dataset: ", df_train.shape[1])


# In[ ]:


df_train.head()


# In[ ]:


df_train.describe()


# In[ ]:


df_train.info()


# #### Test Dataset

# In[ ]:


print("Total number of samples in test dataset: ", df_test.shape[0])
print("Number of columns in test dataset: ", df_test.shape[1])


# In[ ]:


df_test.head()


# #### Sample Submission Dataset

# In[ ]:


df_sample_submission.shape


# In[ ]:


df_sample_submission.head()


# ## 4. Missing Values
# 
# Are there any missing values in the train and test datasets?

# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# _Fortunately, both the train and test datasets are clean & none of the values are missing._

# In[ ]:





# ## 5. Open-Questions/Hypothesis
# (In-Progress)

# ### 5.1. Difference in the columns between the training and test datasets
# The training dataset contains 11 columns and the test dataset contains 9 columns. The two additional columns that are present in the training dataset, and not in the test dataset are dropoff_datetime and trip_duration. By looking at the sample_submission.csv file, we understand that we need to predict the trip_duration.

# ### 5.2. Are all the id's in the train and test datasets unique? Is there an overlap, in the observations, between the train and test datasets?

# In[ ]:


print("Number of ids in the train dataset: ", len(df_train["id"]))
print("Number of unique ids in the train dataset: ", len(pd.unique(df_train["id"])), "\n")

print("Number of ids in the test dataset: ", len(df_test["id"]))
print("Number of unique ids in the test dataset: ", len(pd.unique(df_test["id"])), "\n")

print("Number of common ids(if any) between the train and test datasets: ", len(set(df_train["id"].values).intersection(set(df_test["id"].values))))


# ### 5.3. Are all the vendor_id's in the train and test datasets unique? 
# 
# * vendor_id takes on only two values in both the train and test datasets i.e. 1 and 2 (Hypothesis - This could represent data from two different taxi companies)
# 
# #### This leads to a set of follow-up questions:
# 
# * If the hypothesis is right and the values in the vendor_id column actually represent the data from two different taxi companies; are the number of observations in the dataset from each of the companies comparable or is there any imbalance?(Both in the train and test datasets)
# 
# * Among the vendor_id's(1 and 2) - what is the distribution in the number of passengers (passenger_count) across the train and test datasets?

# In[ ]:


print("Number of vendor_ids in the train dataset: ", len(df_train["vendor_id"]))
print("Number of unique vendor_ids in the train dataset: ", len(pd.unique(df_train["vendor_id"])), "\n")

print("Number of vendor_ids in the test dataset: ", len(df_test["vendor_id"]))
print("Number of unique vendor_ids in the test dataset: ", len(pd.unique(df_test["vendor_id"])), "\n")


# In[ ]:





# In[ ]:


# #The number of observations in the dataset from each of the two companies i.e. 1 and 2, seems to be comparable
# #across the train and test datasets
sns.countplot(x="vendor_id", data=df_train)


# In[ ]:


sns.countplot(x="vendor_id", data=df_test)


# ### 5.4. Distribution of the number of passengers across the vendor_id variables 1 and 2, in both the train and test datasets

# In[ ]:


sns.countplot(x="passenger_count", data=df_train[df_train["vendor_id"] == 1])


# In[ ]:


sns.countplot(x="passenger_count", data=df_train[df_train["vendor_id"] == 2])


# In[ ]:


sns.countplot(x="passenger_count", data=df_test[df_test["vendor_id"] == 1])


# In[ ]:


sns.countplot(x="passenger_count", data=df_test[df_test["vendor_id"] == 2])


# In[ ]:





# ### 5.5. Distribution of the trip_duration across the train dataset

# In[ ]:


# #String to Datetime conversion
df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"])
df_train["dropoff_datetime"] = pd.to_datetime(df_train["dropoff_datetime"])

df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"])


# In[ ]:





# In[ ]:


# #trip_duration represents the difference between the dropoff_datetime and the pickup_datetime in the
# #train dataset
df_train["trip_duration"].describe()


# In[ ]:


# #The trip_duration would be a lot more intuitive when the datetime representation is used, 
# #rather than the representation with seconds. 
(df_train["dropoff_datetime"] - df_train["pickup_datetime"]).describe()


# _It is interesting to see that there happens to be a trip that lasted for over 40 days. Let us plot the trip duration in seconds to view any other possbile outliers._

# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(range(len(df_train["trip_duration"])), np.sort(df_train["trip_duration"]))
plt.xlabel('index')
plt.ylabel('trip_duration in seconds')
plt.show()


# _We see that there are four outliers with trip durations of 20 days or more_

# In[ ]:


# #Removing the outliers in the dataset
df_train = df_train[df_train["trip_duration"] < 500000]


# In[ ]:


(df_train["dropoff_datetime"] - df_train["pickup_datetime"]).describe()


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(range(len(df_train["trip_duration"])), np.sort(df_train["trip_duration"]))
plt.xlabel('index')
plt.ylabel('trip_duration in seconds')
plt.show()


# _It is interesting to see that a lot of trips, have a trip duration nearing 23 hours_

# In[ ]:





# ### 5.6. Distribution of the store_and_fwd_flag across the train dataset

# In[ ]:


sns.countplot(x="store_and_fwd_flag", data=df_train)


# _According to the Data Dictionary the store and fwd flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip. This must therefore indicate that most of the trips(99.448% to be precise) were not stored in the vehicle memory before forwarding._
# 
# _**Hypothesis** - In 99.448% of the trips, the vehicles might have been in an area of NYC, where the celluar reception was good; thereby having a connection to the server. In 0.551% of the trips, the celluar reception might have been poor; thereby having to store the trip record in the vehicle memory prior to sending it to the server. Could this affect the dropoff longitude and latitude? Would the dropoff coordinates not have been recorded until celluar reception was availbale again? If so, must we ignore such records while building the model?_

# In[ ]:


len(df_train[df_train["store_and_fwd_flag"] == "N"])*100.0/(df_train.count()[0])


# _Among the 0.551% of the trips in the train dataset, where the trip record was stored in the vehicle memory; we observe that all the 0.551% of the cases were only for vendor id = 1._

# In[ ]:


set(df_train[df_train["store_and_fwd_flag"] == "Y"]["vendor_id"])


# In[ ]:





# ### 5.7. Exploring the distances between the pickup and dropoff lat/log coordinates

# In[ ]:


from haversine import haversine


# In[ ]:


def calculate_haversine_distance(var_row):
    return haversine((var_row["pickup_latitude"], var_row["pickup_longitude"]), 
                     (var_row["dropoff_latitude"], var_row["dropoff_longitude"]), miles = True)


# In[ ]:


# #Calculating the Haversine Distance
# #The haversine formula determines the great-circle distance between two points on a sphere 
# #given their longitudes and latitudes.
df_train["haversine_distance"] = df_train.apply(lambda row: calculate_haversine_distance(row), axis=1)


# In[ ]:


df_train.head()


# In[ ]:


df_train["haversine_distance"].describe()


# In[ ]:





# _Plot of the haversine distance vs the trip duration._

# In[ ]:


#plt.figure(figsize=(10,10))
#sns.regplot(x="haversine_distance", y="trip_duration", data=df_train)


# ![](http://kartikkannapur.github.io//images/haversine_distance_1.png)

# _The presence of outliers in the train dataset(long tail in the haversine distance) might have caused the deviation in the regression line. It would be interesting to explore different methods for outlier detection._

# In[ ]:


df_train[df_train["haversine_distance"] > 100]


# In[ ]:


#plt.figure(figsize=(10,10))
#sns.regplot(x="haversine_distance", y="trip_duration", data=df_train[df_train["haversine_distance"] < 100])


# ![](http://kartikkannapur.github.io//images/haversine_distance_2.png)

# In[ ]:





# ### 5.8. Exploring the number of trips at each timestamp feature in the train dataset

# _Once the train dataset has been cleaned, based on the outliers in column - trip duration(that consisted of a few trips lasting for 20 days or more), we can now explore the timestamps on a hourly-weekly basis for further exploratory analysis._

# _The train dataset contains trips that range from 2016-01-01 to 2016-06-30, i.e. 6 months worth of data._

# In[ ]:


print("Train dataset start date: ", min(df_train["pickup_datetime"]))
print("Train dataset end date: ", max(df_train["pickup_datetime"]))


# In[ ]:


# #Conversion to pandas to_datetime has already been performed in section 5.5
# #df_train["pickup_datetime"] = pd.to_datetime(df_train['pickup_datetime'])


df_train["pickup_dayofweek"] = df_train.pickup_datetime.dt.dayofweek
df_train["pickup_weekday_name"] = df_train.pickup_datetime.dt.weekday_name
df_train["pickup_hour"] = df_train.pickup_datetime.dt.hour
df_train["pickup_month"] = df_train.pickup_datetime.dt.month


# In[ ]:


df_train.head()


# In[ ]:





# _Distribution of trips across - months in the six month rage, day of the week and hour in a day._
# 
# _We can observe that there are more trips on Friday's and Saturday's, than on any other weekday, and this make sense (TGIF :)); On a 24 hour clock, the number of trips is the highest between 17:00 hrs - 22:00 hrs and reduces post 01:00 hrs; On a six month time range, the number of trips are almost evenly distributed, with none of the months having a surprising spike in the dataset. _

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="pickup_weekday_name", data=df_train)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="pickup_hour", data=df_train)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="pickup_month", data=df_train)
plt.show()


# _It would now be interesting to visualize the behaviour of the trip duration variable, based on the timestamp features._

# In[ ]:





# ### 5.9. Exploring the behaviour of trip_duration based on the number of trips for each timestamp feature in the train dataset

# _In order to visualize the trip duration behaviour, it would be important to aggregate the trip duration at each of the timnestamp feature levels. Since there could be outliers in the trip duration variable(and outlier detection has not yet been performed for this variable) median would be a more representative measure, rather than the mean._

# In[ ]:


df_train.trip_duration.describe()


# In[ ]:


df_train_agg = df_train.groupby('pickup_weekday_name')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.pickup_weekday_name.values, df_train_agg.trip_duration.values)
plt.show()


# In[ ]:


df_train.groupby('pickup_weekday_name')['trip_duration'].describe()


# In[ ]:


df_train_agg = df_train.groupby('pickup_hour')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.pickup_hour.values, df_train_agg.trip_duration.values)
plt.show()


# In[ ]:


df_train.groupby('pickup_hour')['trip_duration'].describe()


# In[ ]:


df_train_agg = df_train.groupby('pickup_month')['trip_duration'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(df_train_agg.pickup_month.values, df_train_agg.trip_duration.values)
plt.show()


# In[ ]:


df_train.groupby('pickup_month')['trip_duration'].describe()


# * Observation at a week-level:
# 
# _Trip durations are the most on Thursday's, Wednesday's and Friday's & the least on Sunday's._
# 
# 
# * Observation at an hour-level:
# 
# _Trip durations are the most between 11:00 hrs and 16:00 hrs & the least between 04:00 hrs and 07:00 hrs._
# 
# * Observation at a month-level:
# 
# _There seems to be a linear increase in the median trip duration from the month of January to the month of June, although the increase is fairly minimal._

# In[ ]:





# ** In Progress  **

# In[ ]:





# ## References
# 
# 1. A Review and Comparison of Methods for Detecting Outliers in Univariate Data Sets by Songwon Seo - http://d-scholarship.pitt.edu/7948/1/Seo.pdf

# In[ ]:




