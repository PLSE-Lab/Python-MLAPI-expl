#!/usr/bin/env python
# coding: utf-8

# # ASHRAE - Great Energy Predictor III
# 
# 
# We predict energy consumption of buildings in this competition.
# 
# Let's see the data to understand the competition.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import gc

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
import seaborn as sns
import random
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc
import tqdm
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# In[ ]:


# Read data...
root = '../input/ashrae-energy-prediction'

train_df = pd.read_csv(os.path.join(root, 'train.csv'))
weather_train_df = pd.read_csv(os.path.join(root, 'weather_train.csv'))
test_df = pd.read_csv(os.path.join(root, 'test.csv'))
weather_test_df = pd.read_csv(os.path.join(root, 'weather_test.csv'))
building_meta_df = pd.read_csv(os.path.join(root, 'building_metadata.csv'))
sample_submission = pd.read_csv(os.path.join(root, 'sample_submission.csv'))


# In[ ]:


train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])
weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])


# # Brief data introduction
# 
# Let's start from looking train data

# In[ ]:


print('train_df shape: ', train_df.shape)
train_df.sample(3)


#  - `building_id`: Foreign key for the building metadata.
#  - `meter`: The meter id code. Read as {0: electricity, 1: chilledwater, 2: steam, hotwater: 3}. Not every building has all meter types.
#  - `timestamp`: When the measurement was taken
#  - `meter_reading`: The target variable. Energy consumption in kWh (or equivalent). Note that this is real data with measurement error, which we expect will impose a baseline level of modeling error.

# Train data contains whole year information in 2016.

# In[ ]:


train_df.iloc[[0,-1]]


# test data has same column of train data, except `meter_reading` which is the target variable.

# In[ ]:


print('test_df shape: ', test_df.shape)
test_df.sample(3)


# test data contains all information from 2017 and 2018.
# 
# So test data ranges longer term than train dataset. You need to estabilish a model to predict far future from only 1 year train data information.

# In[ ]:


test_df.sort_values(by=['timestamp']).iloc[[0, -1]]


# It seems only same 1449 buildings are contained in train and test data.

# In[ ]:


print(f"number of building_id in train {len(train_df['building_id'].unique())}, test {len(test_df['building_id'].unique())}")


# Now let's see submission format.
# 
# sample submission has same row number with test dataset, you need to predict each row of test dataset, and put predicted `meter_reading` number with `row_id`.

# In[ ]:


print('sample_submission.shape', sample_submission.shape)
sample_submission.sample(3)


# Now we understand train/test data and submission format. However any rich information is not in these files.
# 
# Let's look other csv files which stores the feature.

# train and test data contains `building_id` column, you can refer `building_meta_df` for the information of each building.

# In[ ]:


print('building_meta_df shape', building_meta_df.shape)
building_meta_df.sample(3)


#  - `site_id`: Foreign key for the weather files.
#  - `building_id`: Foreign key for training.csv
#  - `primary_use`: Indicator of the primary category of activities for the building based on EnergyStar property type definitions
#  - `square_feet`: Gross floor area of the building
#  - `year_built`: Year building was opened
#  - `floor_count`: Number of floors of the building

# It stores where is the building as `site_id` (which connects to the whether information, we will see next), how big is the building, how old is the building etc.

# In[ ]:


primary_use_list = building_meta_df['primary_use'].unique()
primary_use_list


# `primary_use` key stores 16 type of category that what is the main usage of the building.
# 
# `site_id` is categorized as 16 different area.

# In[ ]:


building_meta_df['site_id'].unique()


# In[ ]:


print('weather_train_df shape', weather_train_df.shape)
weather_train_df.sample(3)


# In[ ]:


print('weather_test_df shape', weather_test_df.shape)
weather_test_df.sample(3)


# Both whether_train and weather_test data contains following weather information, which is unique by `site_id` and `timestamp`.
# 
#  - `site_id`
#  - `air_temperature`: Degrees Celsius
#  - `cloud_coverage`: Portion of the sky covered in clouds, in oktas
#  - `dew_temperature`: Degrees Celsius
#  - `precip_depth_1_hr`: Millimeters
#  - `sea_level_pressure`: Millibar/hectopascals
#  - `wind_direction`: Compass direction (0-360)
#  - `wind_speed`: Meters per second
# 

# That's all for brief data tour. Now let's check each feature in more carefully.

# # Data visualization

# ### meter and meter_reading

# meter category consists of 4 types. 'electricity' is most frequent and 'hotwater' is least frequent.

# In[ ]:


fig, ax = plt.subplots()

test_df.meter.hist(ax=ax, color=[0., 0., 1., 0.5])
train_df.meter.hist(ax=ax, color=[1., 0., 0., 0.5])

# {0: electricity, 1: chilledwater, 2: steam, hotwater: 3}
ax.set_xticks(np.arange(4))
ax.set_xticklabels(['electricity', 'chilledwater', 'steam', 'hotwater'])
plt.show()


# In[ ]:


sns.distplot(train_df['meter_reading']).set_title('meter_reading', fontsize=16)
plt.show()


# It looks very skewed,,, but it's not. We should check `meter_reading` in **log scale**!!
# 
# I use `np.log1p` instead of `np.log`, because it contains 0.0 value.

# In[ ]:


sns.distplot(np.log1p(train_df['meter_reading'])).set_title('meter_reading', fontsize=16)
plt.show()


# Looks normally distributed except 0.0!

# In[ ]:


train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])

titles = ['electricity', 'chilledwater', 'steam', 'hotwater']

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for i in range(4):
    title = titles[i]
    sns.distplot(train_df.loc[train_df['meter'] == i, 'meter_reading_log1p'], ax=axes[i]).set_title(title, fontsize=16)

plt.show()


# In[ ]:


# # takes too much time...
# train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])
# sns.violinplot(x='meter', y='meter_reading_log1p', data=train_df)


# ### timestamp

# It looks timestamp is taken **every hour**.

# In[ ]:


unique_time = train_df['timestamp'].unique()
unique_time[:25], unique_time[-25:]


# ### nan check
# 
# There are many features with nan. How to compensate these incomplete information may be a key to get good performance.

# In[ ]:


print('train_df: total rows', len(train_df))
print('Number of nan')
train_df.isna().sum()


# Most of the building has no `year_built` and `floor_count` information.

# In[ ]:


print('building_meta_df: total rows', len(building_meta_df))
print('Number of nan')
building_meta_df.isna().sum()


# In[ ]:


print('weather_train_df: total rows', len(weather_train_df))
print('Number of nan')
weather_train_df.isna().sum()


# ## meter_reading in time
# 
# We need to predict energy consumption everyday every hour basis. My first assumption is the following.
# 
#  - Energy consumption is high in office hour, while small at night.
#  - Energy consumption is high in weekdays, while small in weekend.
#  - Hot water is more often used in winter, chilled water is used in summer.
#  
# Let's check these assumption is correct or not.

# In[ ]:


# only look specific building
target_building_id = 500
target_meter = 0


# In[ ]:


def plot_meter_reading_in_time(target_building_id, target_meter, target_month):
    target_building_df = train_df[(train_df['building_id'] == target_building_id) & (train_df['meter'] == target_meter)].copy()
    target_building_df['hour'] = target_building_df.timestamp.dt.hour
    target_building_df['month'] = target_building_df.timestamp.dt.month
    target_building_df['day'] = target_building_df.timestamp.dt.day
    target_building_df['dow'] = target_building_df.timestamp.dt.dayofweek

    plt.figure()
    plt.title(f'building_id {target_building_id} meter {target_meter} month {target_month}')
    for day in range(1, 8):
        target_building_df_short = target_building_df[(target_building_df['month'] == target_month) & (target_building_df['day'] == day)]
        plt.plot(target_building_df_short['hour'].values, target_building_df_short['meter_reading_log1p'].values, label=f'day {day}: dow {target_building_df_short.dow.values[0]}')
    plt.legend()
    # plt.scatter(target_building_df['hour'].values, target_building_df['meter_reading_log1p'].values)


# In[ ]:


plot_meter_reading_in_time(target_building_id=300, target_meter=0, target_month=4)
plot_meter_reading_in_time(target_building_id=500, target_meter=0, target_month=6)
plot_meter_reading_in_time(target_building_id=900, target_meter=0, target_month=9)


# I randomly drawed some figure, and it looks electricity (meter=0) tend to be more consumed on office hour (6:00~21:00), and in weekday (dow 0 is Monday, so dow 5 & 6 are weekend).

# In[ ]:


for target_month in range(1, 13, 1):
    plot_meter_reading_in_time(target_building_id=112, target_meter=3, target_month=target_month)


# In above plotting, I checked hot water usage (meter 3) in each month. It looks it is never used during July to October which is summer hot season!

# In[ ]:





# In[ ]:





# In[ ]:





# # Data preprocessing
# 
# Now, Let's try preprocessing to build GBDT (Gradient Boost Decision Tree) model for predicting `meter_reading`.
# 
# Refer [my next kernel](https://www.kaggle.com/corochann/ashrae-simple-lgbm-submission) for the model training part.

# test dataset has 40M rows, we need to care memory before merging other data...!

# In[ ]:


# categorize primary_use column to reduce memory on merge...

primary_use_dict = {key: value for value, key in enumerate(primary_use_list)} 
print('primary_use_dict: ', primary_use_dict)
building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)

gc.collect()


# In[ ]:


from pandas.api.types import is_datetime64_any_dtype as is_datetime

# copy from https://www.kaggle.com/ryches/simple-lgbm-solution
#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not is_datetime(df[col]):  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


# In[ ]:


reduce_mem_usage(train_df)
reduce_mem_usage(test_df)
reduce_mem_usage(building_meta_df)
reduce_mem_usage(weather_train_df)
reduce_mem_usage(weather_test_df)


# In[ ]:


gc.collect()


# In[ ]:


# merge building and weather information to test data...
# be careful that it takes a lot of memory

test_df = test_df.merge(building_meta_df, on='building_id', how='left')
test_df = test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
del weather_test_df
gc.collect()


# In[ ]:


# merge building and weather information to train data...

train_df = train_df.merge(building_meta_df, on='building_id', how='left')
train_df = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

del building_meta_df
del weather_train_df
gc.collect()


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


gc.collect()


# We can load faster by save in feather format, instead of csv. I will use this data in the next [ASHRAE: Simple LGBM submission](https://www.kaggle.com/corochann/ashrae-simple-lgbm-submission) notebook.

# In[ ]:


train_df.to_feather('train.feather')
test_df.to_feather('test.feather')
sample_submission.to_feather('sample_submission.feather')


# In[ ]:


train_df.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # References
# 
# These kernels inspired me to write this kernel, thank you for sharing!
# 
#  - https://www.kaggle.com/rishabhiitbhu/ashrae-simple-eda
#  - https://www.kaggle.com/isaienkov/simple-lightgbm
#  - https://www.kaggle.com/ryches/simple-lgbm-solution

# In[ ]:




