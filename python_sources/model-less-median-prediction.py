#!/usr/bin/env python
# coding: utf-8

#  # Baseline Median Prediction

# 
# The below solution is a simple baseline created by calculating median meter reading values for each building and meter type and then appending them directly to the test set. The idea comes from the assumption that the future meter readings will be similar to the meter readings in the past for each building and meter type.
# 
# Inspiration for this solution comes from the past kaggle competition https://www.kaggle.com/c/web-traffic-time-series-forecasting which was also a time series forecasting problem. There were more than 100,000 separate time series that had to be used to predict future web traffic per page. A few solutions used the median visits over week time periods and one of the high placing solutions used the median predictions as features for modeling.
# 
# https://www.kaggle.com/safavieh/median-estimation-by-fibonacci-et-al-lb-44-9
# https://www.kaggle.com/chechir/weekend-flag-median-with-wiggle
# https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/44729#latest-305761
# 

#  ## What's Old is New Again  
#  
# The solution achieves a public LB of 1.61. This solution is certainly not high performing in the long run but it proves that a simple baseline is a good place to start. For time series forecasting, the most naive assumption for a baseline is that your future values will be the same as what you have seen previously. 
# 
# Data cleaning, feature engineering, feature selection, cross-validation and model selection are as important as ever in this competition but if your model cannot outperform this naive method, you need to rethink your approach.
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# No modeling packages required

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ---

# ### 1. Read in the Data
# In this solution, we will only need the training csv and the test csv as we are only interested in the meter reading values themselves. I will not read in the weather data or building metadata at all.

# In[ ]:


PATH = '/kaggle/input/ashrae-energy-prediction/train.csv'
df = pd.read_csv(PATH)
df.head().append(df.tail())


# From looking at the training data, we can see that we have a year of data in hour increments from 00:00 01/01/2016 to 23:00 12/31/2016. The data represents separate time series for each of the 1448 buildings and the 4 meter types.

# In[ ]:


PATH = '/kaggle/input/ashrae-energy-prediction/test.csv'
test = pd.read_csv(PATH)
test.head().append(test.tail())


# From looking at the testing data, we can see that we are predicting nearly a year and a half of data in hour increments from 00:00 01/01/2017 to 23:00 05/09/2018. In this problem, we are predicting more in the future than we have data for.

# ---

# ### 2. Visualize Each Meter Type
# In addition to the 1448 buildings, the dataset has 4 different types of energy meter: electric, chilled water, steam and hot water. Not every building has every meter type but each building that has more than one type has a different time series for each type of meter. Below are representative visualizations of each of the meter types. Note that each meter type time series has a different pattern, measurement scale and seasonality. This means that we will have to account for the meter type in addition to the building number for our calculation.
# 

# In[ ]:


# Meter 0: Electric 
ts = df.loc[df['building_id']== 206]
ts = ts.loc[ts['meter']== 0]
N = len(ts)
ts['t'] = range(N)
plt.figure(figsize=(30,5))
sns.lineplot(x='t', y='meter_reading', data=ts)
plt.xlabel("Time: 2016-01-01 00:00 - 2016-12-31 23:00 (hr)", size = 12)
plt.ylabel("Meter Reading (kWh)", size = 12)
plt.title('Building 206, Electric Meter', size = 16)
plt.show()


# In[ ]:


# Meter 1: Chilled Water
ts = df.loc[df['building_id']== 206]
ts = ts.loc[ts['meter']== 1]
N = len(ts)
ts['t'] = range(N)
plt.figure(figsize=(30,5))
sns.lineplot(x='t', y='meter_reading', data=ts)
plt.xlabel("Time: 2016-01-01 00:00 - 2016-12-31 23:00(hr)", size = 12)
plt.ylabel("Meter Reading (kWh)", size = 12)
plt.title('Building 206, Chilled Water Meter', size = 16)
plt.show()


# In[ ]:


# Meter 2: Steam
ts = df.loc[df['building_id']== 750]
ts = ts.loc[ts['meter']== 2]
N = len(ts)
ts['t'] = range(N)
plt.figure(figsize=(30,5))
sns.lineplot(x='t', y='meter_reading', data=ts)
plt.xlabel("Time: 2016-01-01 00:00 - 2016-12-31 23:00(hr)", size = 12)
plt.ylabel("Meter Reading (kWh)", size = 12)
plt.title('Building 750, Steam Meter', size = 16)
plt.show()


# In[ ]:


# Meter 3: Hot Water
ts = df.loc[df['building_id']== 206]
ts = ts.loc[ts['meter']== 3]
N = len(ts)
ts['t'] = range(N)
plt.figure(figsize=(30,5))
sns.lineplot(x='t', y='meter_reading', data=ts)
plt.xlabel("Time: 2016-01-01 00:00 - 2016-12-31 23:00 ", size = 12)
plt.ylabel("Meter Reading (kWh)", size = 12)
plt.title('Building 206, Hot Water Meter', size = 16)

plt.show()


# ---

# ### 3. Calculate Median Meter Readings of Training Set
# 
# We will be creating two new features from the timestamp: week and hour. This will allow us to group by building, meter, week and hour to calculate the median.

# In[ ]:


# Set up variables of column names
time = 'timestamp'
target = 'meter_reading'
building = 'building_id'
meter = 'meter'
week = 'week'
hr = 'hour'
pred = 'prediction'
target = 'meter_reading'


# In[ ]:


# Convert time stamp to datetime 
df[time] = pd.to_datetime(df[time])
test[time] = pd.to_datetime(test[time])


# In[ ]:


# Split time stamp into week and hour features
df[week] = df[time].dt.week
df[hr] = df[time].dt.hour

test[week] = test[time].dt.week
test[hr] = test[time].dt.hour

print(df.head())
print(test.head())


# In[ ]:


# Group by building id, meter type, week and hour. Then calculate median value for each group and make a new series called prediction from this. 

group = [building, meter, week, hr]
gp = df.groupby(group)[[target]].median().rename({target:pred},axis=1)


# We can do this because we only have one year of data so we are assuming that the predicted data will be the same as the old data for each week and hour. 
# However, since we are predicting January - May twice in the test set, we will be repeating some of the predicted values.

# In[ ]:


gp.head().append(gp.tail())


# In[ ]:


# Merge the predictions to the test set to align timestamps again
test = test.merge(gp, on=group, how='left')
test


# ### 4. Read in Submission File and Append Predictions

# In[ ]:


# Read in submission file
PATH = '/kaggle/input/ashrae-energy-prediction/sample_submission.csv'
sub = pd.read_csv(PATH)
sub.head().append(sub.tail())


# In[ ]:


# Check that submission ids match test set ids 
# Add in median predictions to submission as target
print(len(sub))
print(len(test))
print((sub['row_id'] == test['row_id']).all())

# Forward fill and back fill nans as a precaution
sub[target] = test[pred].ffill().bfill()


# In[ ]:


sub[[target]]


# In[ ]:


# Write csv for submission
PATH = 'submission.csv'
sub.to_csv(PATH, index=False)


# In[ ]:


sub.head().append(sub.tail())


# In[ ]:


# Final check for nan values in submission
sub.isna().any()


# In[ ]:


np.isinf(sub).any()

