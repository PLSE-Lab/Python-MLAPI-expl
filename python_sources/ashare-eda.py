#!/usr/bin/env python
# coding: utf-8

# # Explaining this notebook
# 
# I would like to explain my EDA for ASHARAE competition.
# 
# I am bigginer of data science.  
# This is my first notebook and I am not good at english.
# 
# So, Please forgive me my poor english.

# # Load Data

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# <pre>
# /kaggle/input/ashrae-energy-prediction/weather_train.csv
# /kaggle/input/ashrae-energy-prediction/test.csv
# /kaggle/input/ashrae-energy-prediction/weather_test.csv
# /kaggle/input/ashrae-energy-prediction/train.csv
# /kaggle/input/ashrae-energy-prediction/building_metadata.csv
# /kaggle/input/ashrae-energy-prediction/sample_submission.csv

# In[ ]:


train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")
building_metadata = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")
weather_train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")

print("train.shape:", train.shape)
print("building_metadata.shape:", building_metadata.shape)
print("weather_train.shape:", weather_train.shape)


# # Merge train data

# In[ ]:


train_merge = train.merge(building_metadata, on= "building_id",how="left")
train_merge = train_merge.merge(weather_train, on=["site_id", "timestamp"], how="left")


# In[ ]:


train_merge.head()


# # make features (day, months, weekdays)

# In[ ]:


train_merge["timestamp"] = pd.to_datetime(train_merge["timestamp"])


# In[ ]:


train_merge["month"] = train_merge["timestamp"].dt.month
train_merge["day"] = train_merge["timestamp"].dt.day
train_merge["weekday"] = train_merge["timestamp"].dt.weekday
train_merge["weekday_name"] = train_merge["timestamp"].dt.weekday_name


# In[ ]:


train_merge.head()


# # calculate outliers

# In[ ]:


def CalcOutliers(df_num): 

    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # seting the cut line to both higher and lower values
    # You can change this value
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print("upper:", upper)
    print("lower:", lower)
    print("-"*30)
    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides
    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points
    
    return


# In[ ]:


CalcOutliers(train_merge["meter_reading"])


# In[ ]:


# detect outlier's building_id

upper = 461823.9846903139
np.unique(train_merge.loc[train_merge["meter_reading"] >= upper, "building_id"].values)


# In[ ]:


# analyzing 778
id_778_outmeter = np.unique(train_merge.loc[(train_merge["meter_reading"] >= upper) & 
                                            (train_merge["building_id"] == 778),"meter"].values)
print("id778 outlier / meter:", id_778_outmeter)

id_778_outmonth = np.unique(train_merge.loc[(train_merge["meter_reading"] >= upper) &
                                            (train_merge["building_id"] == 778), "month"].values)
print("id778 outlier / months:", id_778_outmonth)


for i in id_778_outmonth:
    print("months=", i, sep="")
    print("primary_use", np.unique(train_merge.loc[(train_merge["meter_reading"] >= upper) &
                                        (train_merge["building_id"] == 778) &
                                        (train_merge["month"] == i), "primary_use"].values))
    
    print(np.unique(train_merge.loc[(train_merge["meter_reading"] >= upper) &
                                            (train_merge["building_id"] == 778) &
                                            (train_merge["month"] == i), "day"].values))
    print("-" * 30)


# In[ ]:


# analyzing 1099
id_1099_outmeter = np.unique(train_merge.loc[(train_merge["meter_reading"] >= upper) & 
                                            (train_merge["building_id"] == 1099),"meter"].values)
print("id778 outlier / meter:", id_1099_outmeter)

id_1099_outmonth = np.unique(train_merge.loc[(train_merge["meter_reading"] >= upper) &
                                            (train_merge["building_id"] == 1099), "month"].values)
print("id778 outlier / months:", id_1099_outmonth)

for i in id_1099_outmonth:
    print("month=", i, sep="")
    print("primary_use", np.unique(train_merge.loc[(train_merge["meter_reading"] >= upper) &
                                            (train_merge["building_id"] == 1099) &
                                            (train_merge["month"] == i), "primary_use"].values))
    
    print("days", np.unique(train_merge.loc[(train_merge["meter_reading"] >= upper) &
                                            (train_merge["building_id"] == 1099) &
                                            (train_merge["month"] == i), "day"].values))
    print("-" * 30)


# Building_id 778 and 1099 have outliers.  
# 
# Outliers of 778 is in september and october.  
# Outliers of 1099 is in Jan, Feb, Mar, Apl, May, Jun and Nov.
# 
# I can not detect rule of days that have outliers.

# # EDA

# In[ ]:


train_merge.groupby(["timestamp"])["meter_reading"].mean().plot(figsize=(20, 5))


# I feel strange because this chart has plot that suddenly decrease.

# In[ ]:


train_merge.groupby(["month"])["meter_reading"].mean().plot(kind="bar", figsize=(20, 5))


# Peak is Apl  
# After Jul, "meter reading" is low level.

# In[ ]:


train_merge.groupby(["day"])["meter_reading"].mean().plot(kind="bar", figsize=(20, 5))


# In[ ]:


train_merge.groupby(["air_temperature"])["meter_reading"].mean().plot(figsize=(20, 5))


# Low temperature has high "meter reading"  
# 
# So, I think that, cold has more "meter reading" than hot.  
# In my opinion, high "wind_speed" has more "meter_reading"

# In[ ]:


train_merge.groupby(["wind_speed"])["meter_reading"].mean().plot(figsize=(20, 5))


# But, it is not related

# In[ ]:


train_merge.groupby(["weekday_name"])["meter_reading"].mean().plot(kind="bar", figsize=(20, 5))


# Weekends have low level "meter reading".  
# ![](http://)So, "Weekday" has possible to important feature.

# In[ ]:




