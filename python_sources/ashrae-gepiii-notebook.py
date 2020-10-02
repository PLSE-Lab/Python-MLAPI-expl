#!/usr/bin/env python
# coding: utf-8

# My notebook for the [ASHRAE Great Energy Predictor III](https://www.kaggle.com/c/ashrae-energy-prediction) competition.

# # Import libraries

# In[ ]:


# General libs
import os
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML/DL libs
# from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, SGDRegressor
# from sklearn import metrics
import tensorflow as tf
from tensorflow import keras

# Other stuff
import pickle

# Constants
root = '../input/ashrae-energy-prediction/'


# # Data Exploration and Imputation

# In[ ]:


# TODO: compartmentalize this code so it's callable for both test and train data


# ### Training data

# In[ ]:


type_dict_meter = {  # Change data types to reduce size and training time.
    'building_id': 'int16',
    'meter': 'int8',
    'meter_reading': 'float32'}
train_df = pd.read_csv(
    os.path.join(root, 'train.csv'),
    dtype=type_dict_meter,
    parse_dates=['timestamp'])
train_df.describe()


# There is an additional column called timestamp. None of the columns are missing data.

# ### Weather training data

# In[ ]:


type_dict_weather = {  # Change data types to reduce size and training time.
    'site_id': 'int8',
    'air_temperature': 'float32',
    'cloud_coverage': 'float32',
    'dew_temperature': 'float32',
    'precip_depth_1_hr': 'float32',
    'sea_level_pressure': 'float32',
    'wind_direction': 'float32',
    'wind_speed': 'float32'}
wtrain_df = pd.read_csv(
    os.path.join(root, 'weather_train.csv'),
    dtype=type_dict_weather,
    parse_dates=['timestamp'])
wtrain_df.describe()


# Looks like there is a significant amount of data missing from the weather training set. We'll have to perform some inference to fill this in. Ignoring the missing data is not useful as it comprises a significant portion of the dataset. We'll need to infer using time-series methods. Use this: https://www.kaggle.com/juejuewang/handle-missing-values-in-time-series-for-beginners. Fortunately, this data is already sorted by site, then timestamp, meaning that we can easily interpolate without sorting.
# 
# Note: cloud_coverage is in [oktas](https://en.wikipedia.org/wiki/Okta) which notate 1/8ths of the sky covered. A "9" means no data/sky not observable, which is "usually due to dense fog or snow", which implies heavy cloud coverage).
# 
# We also have a timestamp column, which is not missing any data.
# 
# Data that is missing completely by site:
# * 1: precip_depth_1_hr
# * 5: precip_depth_1_hr, sea_level_pressure
# * 7: cloud_coverage
# * 11: cloud_coverage
# * 12: precip_depth_1_hr

# In[ ]:


# Fill in data that's entirely missing from some sites.
# cloud_coverage as -1 as it's categorical
# precip_depth_1_hr and sea_level_pressure as column mean
wtrain_df.loc[wtrain_df['site_id'].isin([7,11]), 'cloud_coverage'] = wtrain_df.loc[wtrain_df['site_id'].isin([7,11]), 'cloud_coverage'].fillna(-1)
wtrain_df.loc[wtrain_df['site_id'].isin([1,5,12]), 'precip_depth_1_hr'] = wtrain_df.loc[wtrain_df['site_id'].isin([1,5,12]), 'precip_depth_1_hr'].fillna(wtrain_df['precip_depth_1_hr'].mean())
wtrain_df.loc[wtrain_df['site_id'].isin([5]), 'sea_level_pressure'] = wtrain_df.loc[wtrain_df['site_id'].isin([5]), 'sea_level_pressure'].fillna(wtrain_df['sea_level_pressure'].mean())


# In[ ]:


# Perform linear interpolation for all items.
# Manual review of the data looks like several sites just measure cloud coverage
# once every few hours, which indicates that it probably doesn't change that fast.
wtrain_df.interpolate(method='linear', inplace=True)
wtrain_df.fillna(-1, inplace=True)  # linear method doesn't fill start/end values
# Reference: https://medium.com/@drnesr/filling-gaps-of-a-time-series-using-python-d4bfddd8c460


# In[ ]:


# Convert cloud_coverage to int as it should be a category.
wtrain_df['cloud_coverage'] = wtrain_df['cloud_coverage'].astype('int8')


# ### Building Metadata

# In[ ]:


type_dict_bm = {  # Change data types to reduce size and training time.
    'site_id': 'int8',
    'building_id': 'int16',
    'square_feet': 'float32',
    'year_built': 'float32',
    'floor_count': 'float32'}
bm_df = pd.read_csv(
    os.path.join(root, 'building_metadata.csv'),
    dtype=type_dict_bm)
bm_df['primary_use'] = pd.Categorical(bm_df['primary_use']).codes
bm_df.describe()


# This set is also missing a lot of data. Looking at the data externally, floor_count seems to toggle by site (either all buildings at a site_id have a floor_count or none do), except for site 12. I'm not sure if that's significant. However, I can predict the floor_count using the square_feet since they are correlated. If I wanted to be more precise, I could also group by primary_use.
# 
# Similarly, the year_built is missing for several rows. However, its presence does not seem to toggle with other columns. There are no years listed before 1900, though it seems unlikely that 774 of the 1449 buildings were built before that year. This data also seems too important to ignore as more modern building methods should influence the overall power usage. I will likely just fill with the mean.

# In[ ]:


# Perform floor_count imputation using linear regression model based on average of each floor_count category.
avg_sq_feet = bm_df.groupby(['floor_count']).mean()  # Will contain average square_feet for each floor_count
sns.regplot(x=avg_sq_feet['square_feet'], y=avg_sq_feet.index.to_series())
plt.show()


# In[ ]:


# Dropping these two floor_counts as they only have 1 or 2 buildings each, and
# regressing with them causes predictions for low square_feet values to be off.
avg_sq_feet.drop([13, 16], inplace=True)

# Create regression model.
floor_reg = LinearRegression()
floor_reg.fit(
    avg_sq_feet['square_feet'].values.reshape(-1, 1),  # x
    avg_sq_feet.index.values.reshape(-1, 1))  # y

# Fill df with predicted values.
for index, row in bm_df.iterrows():
    if np.isnan(row['floor_count']):
        bm_df.at[index, 'floor_count'] = round(floor_reg.predict(np.array([[row['square_feet']]]))[0][0])


# In[ ]:


# Look at values after filling - should be pretty linear.
avg_sq_feet = bm_df.groupby(['floor_count']).mean()
sns.scatterplot(x=avg_sq_feet['square_feet'], y=avg_sq_feet.index.to_series())
plt.show()


# In[ ]:


# Lazy imputation for year_built - we'll use the overall mean.
bm_df['year_built'].fillna(round(bm_df['year_built'].mean()), inplace=True)


# In[ ]:


# Modify site 0 electric meter readings due to
# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261#latest-682635
# meter 0 is the electricity
site_0_buildings = bm_df[bm_df['site_id']==0]['building_id']
train_df.loc[(train_df['building_id'].isin(site_0_buildings)) & (train_df['meter']==0), 'meter_reading'] *= 0.2931


# # Feature Engineering

# In[ ]:


# TODO
# Align temp by (hour) timestamps to local time: https://www.kaggle.com/nz0722/aligned-timestamp-lgbm-by-meter-type
# Do for weather data, do not for meter data (check each first)


# In[ ]:


# Add derived columns from timestamp.
# year not useful as all data is in 2016, and day not as useful as month for seasonal trends.
wtrain_df['month'] = wtrain_df['timestamp'].map(lambda x: x.month).astype('int8')
wtrain_df['hour'] = wtrain_df['timestamp'].map(lambda x: x.hour).astype('int8')
wtrain_df['weekday'] = wtrain_df['timestamp'].map(lambda x: x.dayofweek).astype('int8')
wtrain_df['weekend'] = wtrain_df['weekday'].map(lambda x: 1 if x in [5,6] else 0).astype('bool')  # 5/6 are Sat/Sun


# # Merge Datasets
# 
# We need to merge the weather, meter, and building data into a form usable for training. We need to join the train_df (meter data) with bm_df by building_id, and join the train_df with wtrain_df by timestamp and site_id. We also need to extract the meter readings as our output variable, and delete our unused dataframes.

# In[ ]:


train_df = pd.merge(left=train_df, right=bm_df, on='building_id')
train_df = pd.merge(left=train_df, right=wtrain_df, on=['site_id', 'timestamp'])
train_df.describe()


# In[ ]:


meter_reading = train_df.pop('meter_reading')  # Pop meter_reading off for training.
train_df.drop('timestamp', axis=1, inplace=True)  # Remove timestamp.


# In[ ]:


del wtrain_df


# # Model Selection and Training

# ### Model Selection
# 
# What type of model will work best for this data? We have a millions of data points, and a few dozen features at most. Predicted outputs must be quantitative, not categorical.
# 
# Given these requirements, [my favorite lazy method to decide on a model](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) suggests using a [simple SGD regressor](https://scikit-learn.org/stable/modules/sgd.html). If this model gives poor performance, then I'll try a neural network with simple structure.

# ### Create Model

# In[ ]:


# Perform feature scaling prior to training SGD regressor.
feat_scal_dict = {}
for col in train_df.columns:
    
    if train_df[col].dtypes in ['int8', 'int16', 'int32', 'bool']:
        continue
    
    t_max = train_df[col].max()
    t_min = train_df[col].min()
    diff = t_max - t_min
    feat_scal_dict[col] = (t_max, t_min, diff)
    train_df[col] -= t_min
    train_df[col] /= diff


# In[ ]:


model_sgd = SGDRegressor(  # Currently set to library defaults.
    loss='squared_loss',  # options: [squared_loss, huber, epsilon_insensitive, squared_epsilon_insensitive]
    # epsilon=1  # for [huber, epsilon_insensitive, squared_epsilon_insensitive] models
    max_iter=1000,
    alpha=1e-4,
    tol=1e-3,
    # learning_rate='constant',  # Default is invscaling
    eta0=1e0,  # Default is 1e-2
    # power_t=0.9,  # Default is 0.5
    early_stopping=False,
    verbose=1)  # Default is 0


# ### Train Model

# In[ ]:


model_sgd.fit(train_df, meter_reading)


# ### Save Model

# In[ ]:


pickle.dump(model_sgd, open('model_sgd.model', 'wb'))
# To load model:
# model_sgd = pickle.load(open(filename, 'rb'))


# # Load and Modify Test Data

# ### Test data

# In[ ]:


# TODO


# ### Weather test data

# In[ ]:


# TODO


# # Make Predictions

# In[ ]:


# TODO


# In[ ]:


# TODO: for SGD model, undo feature scaling.
# use feat_scal_dict


# In[ ]:


# Modify site 0 electric meter readings due to
# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261#latest-682635
# meter 0 is the electricity
# TODO: convert back for submission/scoring (multiply by 3.4118)


# # References
# 
# Some other notebooks I've referenced:
# * https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113338#latest-667617
#     * site_id likely corresponds to ASHRAE regions (there are 16); however may not be useful as some sites cover very large geographic areas
#     * ASHRAE regions: https://www.ashrae.org/communities/regions
# * https://www.kaggle.com/c/ashrae-energy-prediction/discussion/114483#latest-670519
#     * this guy is smart
# * https://www.kaggle.com/patrick0302/locate-cities-according-weather-temperature
#     * potential site locations
# * https://www.kaggle.com/hamditarek/reducing-memory-size-for-great-energy-predictor
#     * reducing memory size by 60%

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.

