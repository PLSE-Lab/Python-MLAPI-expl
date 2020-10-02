#!/usr/bin/env python
# coding: utf-8

# # ASHRAE - Great Energy Predictor III

# In this competition our task is to estimate the energy consumption of buildings over time. Significant investments are being made to improve building efficiencies to reduce costs and emissions but current methods of estimation are fragmented and do not scale well. In this competition, we have to develop accurate predictions of metered building energy usage in the following areas: chilled water, electric, natural gas, hot water, and steam meters. The data comes from over 1,000 buildings over a three-year timeframe. 

# In[ ]:


get_ipython().system('ls -lh ../input/ashrae-energy-prediction/')


# In[ ]:


import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook

import altair as alt
from altair.vega import v5
from IPython.display import HTML

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('figure', figsize=(15.0, 8.0))


# In[ ]:


get_ipython().run_cell_magic('time', '', "root = '../input/ashrae-energy-prediction/'\ntrain_df = pd.read_csv(root + 'train.csv')\nweather_train_df = pd.read_csv(root + 'weather_train.csv')\ntest_df = pd.read_csv(root + 'test.csv')\nweather_test_df = pd.read_csv(root + 'weather_test.csv')\nbuilding_meta_df = pd.read_csv(root + 'building_metadata.csv')\nsample_submission = pd.read_csv(root + 'sample_submission.csv')")


# ### Let's take a brief overview of the train and test data

# In[ ]:


train_df.head()


# * `building_id` - Foreign key for the building metadata.
# * `meter` - The meter id code. Read as {0: electricity, 1: chilledwater, 2: steam, hotwater: 3}. Not every building has all meter types.
# * `timestamp` - When the measurement was taken
# * `meter_reading` - **The target variable.** Energy consumption in kWh (or equivalent). Note that this is real data with measurement error, which we expect will impose a baseline level of modeling error.

# In[ ]:


print(f'There are {train_df.shape[0]} rows in train data.')
print(f"There are {train_df['meter'].nunique()} distinct meters in train data.")
print(f"There are {train_df['building_id'].nunique()} distinct buildings in train data, same as the number of rows in building_meta data")


# In[ ]:


building_meta_df.head()


# `building_meta.csv` contains meta features for each building present in the dataset, the attributes definitions are as follows:
# * `site_id` - Foreign key for the weather files.
# * `building_id` - Foreign key for training.csv
# * `primary_use` - Indicator of the primary category of activities for the building based on EnergyStar property type definitions
# * `square_feet` - Gross floor area of the building
# * `year_built` - Year building was opened
# * `floor_count` - Number of floors of the building
# 

# In[ ]:


print(f'There are {building_meta_df.shape[0]} rows in building meta data.')


# In[ ]:


weather_train_df.head()


# `weather_[train/test].csv` contains weather data from a meteorological station as close as possible to the site.
# 
# * `site_id`
# * `air_temperature` - Degrees Celsius
# * `cloud_coverage` - Portion of the sky covered in clouds, in oktas
# * `dew_temperature` - Degrees Celsius
# * `precip_depth_1_hr` - Millimeters
# * `sea_level_pressure` - Millibar/hectopascals
# * `wind_direction` - Compass direction (0-360)
# * `wind_speed` - Meters per second

# In[ ]:


print(f'There are {weather_train_df.shape} rows in weather train data.')


# #### Let's take a look at the test data

# In[ ]:


test_df.head()


# `test.csv`
# 
# The submission files use row numbers for ID codes in order to save space on the file uploads. test.csv has no feature data; it exists so you can get your predictions into the correct order.
# 
# * `row_id` - Row id for your submission file
# * `building_id` - Building id code
# * `meter` - The meter id code
# * `timestamp` - Timestamps for the test data period

# In[ ]:


print(f'There are {test_df.shape[0]} rows in test data.')
print(f"There are {test_df['meter'].nunique()} distinct meters in test data.")
print(f"There are {test_df['building_id'].nunique()} distinct buildings in test data.")


# In[ ]:


sorted(train_df['building_id'].unique()) == sorted(test_df['building_id'].unique())


# So, the we have 1449 `building_id`s in train and test data and both share the same set of ids and meta features as present in `building_meta.csv`

# In[ ]:


weather_test_df.head()


# `weather_test_df` has same attribute definitions as that of `weather_train_df`

# In[ ]:


print(f'There are {weather_test_df.shape} rows in weather test data.')


# In[ ]:


sample_submission.head()


# `row_id` is a foreign_key to `test.csv`, we have to predict `meter_reading` for corresponding row in the test.csv

# In[ ]:


print(f'There are {sample_submission.shape} rows in sample submission file.')


# # Let's do some deeper analysis

# In[ ]:


train_df = train_df.merge(building_meta_df, on='building_id', how='left')
train_df = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')


# In[ ]:


train_df.describe()


# Let's see `meter` counts

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.countplot(y="meter", data=train_df, palette=['navy', 'darkblue', 'blue', 'dodgerblue']).set_title('Meter count', fontsize=16)
plt.yticks(fontsize=14)
plt.xlabel("Count", fontsize=15)
plt.ylabel("Meter number", fontsize=15)
plt.show(plot)


# We can see that meter frequency is in order of 0 (electricity) > 1 (chilledwater) > 2 (steam) > 3 (hotwater)

# Let's analyse meter readings with histograms and violin plots:

# In[ ]:


fig, ax = plt.subplots(figsize = (18, 6))
plt.subplot(1, 2, 1);
plt.hist(train_df['meter_reading']);
plt.title('Basic meter_reading histogram');
plt.subplot(1, 2, 2);
sns.violinplot(x='meter', y='meter_reading', data=train_df);
plt.title('Violinplot of meter_reading by meter');


# Few interesting inferences:
# 
# * Almost all of meter_reading values are close to zero
# * meter_readings from meter number 2 is exceptionally higher compared to other meters

# In[ ]:


train_df.query('meter_reading==0').shape, train_df.shape


# So, about 10% of meter_readings are zero, let's plot `meter_readings` by `meter` distribution plot one by one.

# In[ ]:


def plot_dist(meter, clip_from):
    '''Plots distribution of non zero train data vs meter number''' 
    df = train_df.query(f'meter=={meter} and meter_reading!=0')
    fig, ax = plt.subplots(figsize = (18, 6))
    plt.subplot(1, 3, 1);
    plt.hist(df['meter_reading']);
    plt.title('Basic meter_reading histogram');
    plt.subplot(1, 3, 2);
    sns.violinplot(x='meter', y='meter_reading', data=df);
    plt.title('Violinplot of meter_reading by meter');
    plt.subplot(1, 3, 3);
    sns.violinplot(x='meter', y='meter_reading', data=df.query(f'meter_reading < {clip_from}'));
    plt.title('*Clipped* Violinplot of meter_reading by meter');


# In[ ]:


plot_dist(0, 2000)


# In[ ]:


plot_dist(1, 10000)


# In[ ]:


plot_dist(2, 10000)


# In[ ]:


plot_dist(3, 2000)


# Let's look at the correlation plot

# In[ ]:


# Compute the correlation matrix
corr = train_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# We can see meter_reading has low correlation with other attributes, hmm ..

# ### Let's try to find out the trend and seasonality in meter_readings

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[ ]:


building_id = 400
building = train_df.query(f'building_id == {building_id} and meter==0')


# In[ ]:


decomposition = seasonal_decompose(building['meter_reading'].values, freq=24)
decomposition.plot();


# You can try out different building ids to see that the seasonality is not that significant in many cases.

# Work in progress..

# ### Let's analyse the train/test weather data

# In[ ]:


weather_train_df.columns


# In[ ]:


def plot_kde(column):
    plot = sns.jointplot(x=train_df[column][:10000], y=train_df['meter_reading'][:10000], kind='kde', color='blueviolet')
    plot.set_axis_labels('meter', 'meter_reading', fontsize=16)
    plt.show()


# In[ ]:


def plot_dist_col(column):
    '''plot dist curves for train and test weather data for the given column name'''
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.distplot(weather_train_df[column].dropna(), color='darkorange', ax=ax).set_title(column, fontsize=16)
    sns.distplot(weather_test_df[column].dropna(), color='purple', ax=ax).set_title(column, fontsize=16)
    plt.xlabel(column, fontsize=15)
    plt.legend(['train', 'test'])
    plt.show()


# In[ ]:


plot_dist_col('air_temperature')


# In[ ]:


plot_dist_col('cloud_coverage')


# In[ ]:


plot_dist_col('dew_temperature')


# In[ ]:


plot_dist_col('precip_depth_1_hr')


# In[ ]:


plot_dist_col('sea_level_pressure')


# In[ ]:


plot_dist_col('wind_direction')


# In[ ]:


plot_dist_col('wind_speed')


# Conclusion: the train and test weather distributions are quite similar

# ### Let's analyse the building meta data

# In[ ]:


building_meta_df.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
plot = sns.countplot(y="primary_use", data=building_meta_df, 
                     palette='YlGn').set_title('Primary category of activities for the building', fontsize=16)
plt.yticks(fontsize=14)
plt.xlabel("Count", fontsize=15)
plt.ylabel("Building use type", fontsize=15)
plt.show(plot)


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.distplot(building_meta_df['square_feet'], color='indigo', 
             ax=ax).set_title('Gross floor area of the building', fontsize=16)
plt.xlabel('square_feet', fontsize=15)
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.distplot(building_meta_df['year_built'].dropna(), color='crimson', 
             ax=ax).set_title('Year building was opened', fontsize=16)
plt.xlabel('year_built', fontsize=15)
plt.show()


# All buildings were built between 1990 and 2017

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.distplot(building_meta_df['floor_count'].dropna(), color='crimson', 
             ax=ax).set_title('Number of floors of the building', fontsize=16)
plt.xlabel('floor_count', fontsize=15)
plt.show()


# Work in Progress ..

# <font size=4 color='red'>Do upvote if you liked this kernel :)</font>

# ## References

# * https://www.kaggle.com/tarunpaparaju/lyft-competition-understanding-the-data
# * https://www.kaggle.com/artgor/molecular-properties-eda-and-models
