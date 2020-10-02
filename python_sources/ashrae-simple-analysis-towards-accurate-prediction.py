#!/usr/bin/env python
# coding: utf-8

# This kernel analyses the given ASHRAE data vis-a-vis features to be used for the desired accurate prediction.
# 
# **Upvote the kernel if found useful**

# In[ ]:



import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import xgboost as xgb
import plotly.express as px
from IPython.display import display
from datetime import datetime
import sklearn.metrics
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
weather_train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')


# What does the data look like?

# In[ ]:


train_df.head()


# In[ ]:


train_df.dtypes


# The "timestamp" has to be corced/cast as datetime.

# In[ ]:


train_df['timestamp'] = train_df['timestamp'].astype('datetime64[ns]')


# Having  quick glimpse at the "weather_train" file stored in weather_train_df

# In[ ]:


weather_train_df.head(3)


# Similarly, the "timestamp" field in weather_train_df is converted to datatetime type.

# In[ ]:


weather_train_df['timestamp'] = weather_train_df['timestamp'].astype('datetime64[ns]')


# In[ ]:


print('lengths of data in train and weather_train file are:{tr} and {we} respectively.'.format(tr = train_df.shape[0], we = weather_train_df.shape[0]))


# Similar to the training files, we retrieve the test files and cast the "timestamp" field as datetime

# In[ ]:


test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')
weather_test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')
test_df['timestamp'] = test_df['timestamp'].astype('datetime64[ns]')
weather_test_df['timestamp'] = weather_test_df['timestamp'].astype('datetime64[ns]')


# Let's have a glimpse of the building metadata file

# In[ ]:


bld_data_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
bld_data_df.head(3)


# In[ ]:


bld_data_df.dtypes


# Merging the dataframes, it will be noticed that the key between "train_df" and "bld_data_df" is 'building_id'. The resulting dataframe can then be joined with weather_train_df on'site_id' and 'timestamp' to get the corresponding weather conditions for the energy measurements.

# In[ ]:


train_data_df = train_df.copy()
train_data_df = train_data_df.merge(bld_data_df, on='building_id', how='left')
train_data_df = train_data_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')


# Note that the merge command automatically eleminate duplicate columns in the resulting dataframe (train_data_df).
# 
# A quick check here.........

# In[ ]:


train_data_df.columns


# Let's have a glimpse of the combined training dataset

# In[ ]:


train_data_df.head()


# Obviously, some entries are NaN (null). While we are less concerned about their origin (e.g. data not collected, get corrupted etc), Such entries need to be handled. A lazy way is to drop all rows with one or more NaN values in the training dataset. However, this luxury is impossible in testing, hence, rendering such an option less attractive but not uncommon. Handling missing data is no doubt well documented in research. A concise introduction to handling missing values can be found [here](https://www.kaggle.com/juejuewang/handle-missing-values-in-time-series-for-beginners).
# 
# Let's have a quick glimpse of the data that is null for each field

# In[ ]:


null_checks = pd.concat([train_data_df.isnull().sum(),train_data_df.isnull().sum()/train_data_df.isnull().count()*100],axis=1, keys = ['Total no of null entries','Percentage of null entries'])


# In[ ]:


null_checks.sort_values(by='Percentage of null entries', ascending = False)


# The 'floor_count' field has the highest percentage of missing  entries. Using any type of interpolation to fill in the missing values will likely introduce more uncertainties into the prediction proportional to the percentage of the missing values. No doubt that the number of floors in a building could influence its overall energy consumption. However, imagine that we want to lean on 17.35% of known cases to determine non-analytically the values of the unknown 86.5%, how accurate will tha be?
# 
# The same argument may be applied to the 'year_built' field. For the other 7 fields where the ratios of missing values are less than 50%, it might be tenable to explore a careful method to interpolate the missing entries.
# 
# This will be revisited shortly during feature selection. Let's consider what can cause "fake imbalance" within the dataset i.e. outliers!

#  

# ### OUTLIERS......
# A very nice analysis of the dataset to detect outliers can be found [here](https://www.kaggle.com/juanmah/ashrae-outliers/notebook). Outliers no doubt create what I call "fake imbalance" within a dataset. Their adverse impact on a machine learning model performance cannot be taken for granted.
# 
# First, let's visualise the distribution of the average meter reading across all buildings over time.
# 

# In[ ]:


mean_meter_reading =  train_data_df.groupby('timestamp')['meter_reading'].mean()


# In[ ]:


mean_meter_reading.plot(figsize=(14,8))


# Erratic patterns displayedby the average energy consumptions raise some valid concerns, prompting further investigations. While low energy consumption during peak summer is not uncommon in some regions, the spike in the mean between Nov and Dec is very suspicious.
# 
# Let's aggregate on the 'primary_use' field to see which type has highest average over the entire period

# In[ ]:


primary_use_agg_meter = train_data_df.groupby(['primary_use']).agg({'meter_reading':['count','sum', 'idxmax', 'max']})


# In[ ]:


primary_use_agg_meter


# 1. The output of the aggregate above has multilevel index and it may be better to reconstruct the column names. A function is created to do this as this will likely be used more than once along the line.

# In[ ]:


def reshape_agg_dataframe(agg_dataframe):
    level_0 = agg_dataframe.columns.droplevel(0)
    level_1 = agg_dataframe.columns.droplevel(1)
    level_0 = ['' if x == '' else '-' + x for x in level_0]
    agg_dataframe.columns = level_1 + level_0
    agg_dataframe.rename_axis(None, axis=1)
    return agg_dataframe


# In[ ]:


primary_use_agg_meter = reshape_agg_dataframe(primary_use_agg_meter)


# In[ ]:


primary_use_agg_meter.head(2)


# In[ ]:


primary_use_agg_meter.sort_values(by='meter_reading-sum', ascending= False)


# Education centres have the highest average energy consumption. This itself may not be a problem if the individual building has reasonable average consumption compared to others like indsutrial or technology centres. Also, highest individual comsumption is associated to an education building.

# In[ ]:


train_data_df.iloc[8907488,:]


# In[ ]:


train_data_df['meter_type']= pd.Categorical(train_data_df['meter']).rename_categories({0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'})
daily_train = train_data_df.copy()
daily_train['date'] = daily_train['timestamp'].dt.date
del daily_train['meter']
daily_train = daily_train.groupby(['date', 'building_id', 'meter_type']).sum()
daily_train


# In[ ]:


daily_train_agg = daily_train.groupby(['date', 'meter_type']).agg({'meter_reading':['sum', 'mean', 'idxmax', 'max']})
daily_train_agg.head()


# In[ ]:


daily_train_agg = daily_train_agg.reset_index()
daily_train_agg = reshape_agg_dataframe(daily_train_agg)
daily_train_agg.head(3)


# ### Checking energy consumption across each meter type
# 
# **Electricity**

# In[ ]:


def show_figure(df,x_val, y_val,color_val, title_val):
    fig = px.line(df, x=x_val, y=y_val, color=color_val, render_mode='svg')
    fig.update_layout(title = title_val)
    fig.show(figsize=(16,12))
    return


# In[ ]:


show_figure(daily_train_agg,x_val='date',y_val='meter_reading-sum',color_val='meter_type',title_val='Total kWh per energy aspect')


# In[ ]:


daily_train_agg['building_id_max'] = [x[1] for x in daily_train_agg['meter_reading-idxmax']]
daily_train_agg.head()


# In[ ]:


print('Number of days that a building has the maximum electricity consumption of all the buildings:\n')
print(daily_train_agg[daily_train_agg['meter_type'] == 'electricity']['building_id_max'].value_counts())


# The max values for electricity type are caused by only 6 buildings.

# In[ ]:


daily_train_electricity = daily_train_agg[daily_train_agg['meter_type']=='electricity'].copy()
daily_train_electricity['building_id_max'] = pd.Categorical(daily_train_electricity['building_id_max'])
show_figure(daily_train_electricity,x_val='date',y_val='meter_reading-max',color_val='building_id_max',title_val='Maximum consumption values for the day and energy aspect')


# **Chilledwater**

# In[ ]:


print('Number of days that a building has the maximum chilledwater consumption of all the buildings:\n')
print(daily_train_agg[daily_train_agg['meter_type'] == 'chilledwater']['building_id_max'].value_counts())


# In[ ]:


daily_train_chilledwater = daily_train_agg[daily_train_agg['meter_type']=='chilledwater'].copy()
daily_train_chilledwater['building_id_max'] = pd.Categorical(daily_train_chilledwater['building_id_max'])
show_figure(daily_train_chilledwater,x_val='date',y_val='meter_reading-max',color_val='building_id_max',title_val='Maximum consumption values for the day and energy aspect')


# **Steam**

# In[ ]:


print('Number of days that a building has the maximum steam consumption of all the buildings:\n')
print(daily_train_agg[daily_train_agg['meter_type'] == 'steam']['building_id_max'].value_counts())


# In[ ]:


daily_train_steam = daily_train_agg[daily_train_agg['meter_type']=='steam'].copy()
daily_train_steam['building_id_max'] = pd.Categorical(daily_train_steam['building_id_max'])
show_figure(daily_train_steam,x_val='date',y_val='meter_reading-max',color_val='building_id_max',title_val='Maximum consumption values for the day and energy aspect')


# **Hotwater**

# In[ ]:


print('Number of days that a building has the maximum hotwater consumption of all the buildings:\n')
print(daily_train_agg[daily_train_agg['meter_type'] == 'hotwater']['building_id_max'].value_counts())


# In[ ]:


daily_train_hotwater = daily_train_agg[daily_train_agg['meter_type']=='hotwater'].copy()
daily_train_hotwater['building_id_max'] = pd.Categorical(daily_train_hotwater['building_id_max'])
show_figure(daily_train_hotwater,x_val='date',y_val='meter_reading-max',color_val='building_id_max',title_val='Maximum consumption values for the day and energy aspect')


# These abberant behaviours can be caused by various reasons such as: faulty meter; mistakes in taking the readings; langiage culture representation for numeric values (comma vs dot) etc.
# 
# While some useful information might be lost,data for the buildings identified as outliers in the analysis above could be excluded from training the machine learning model. Moreover, there are just a handful of them.
# 
# Another way to supress the effect will be to normalise the meter reading across each building between, for instance, -1 and 1 or 0 and 1.

# In[ ]:


show_figure(daily_train_hotwater,x_val='date',y_val='meter_reading-max',color_val='building_id_max',title_val='Maximum consumption values for the day and energy aspect')


# In[ ]:


train_data_df.groupby('building_id').building_id.unique()


# Deep dive into the final business..

# In[ ]:


train_data_df_m = train_data_df.copy()


# In[ ]:


train_data_df_m.count()


# ### Selection of Correlating Features
# 
# Intuitively, which fields should be included and which shouldn't be included?
# 
# Building_id: Should this have influence on the energy consumption of a building? Does it matter if a building has an id of '4000' or '00004' or '1'? Will that have any effect on the energy consumption of a building?
# The same argument applies to site_id
# 
# Weather condition (you may want o read [this](https://sciencing.com/air-movement-affect-weather-8657368.html): there is no doubt that the atmospheric temperature affects the energy consumption of a building. Literally, the colder it is, the more the required heating. Conversely, it could also be argued tha the hotter it is, the more the energy required for colling.
# 
# In essence, one will never have to feed every single field of given fields data into building a machine learning. Using rule of thumb to knock off some fields may be good idea even before trying any intelligent feature selection process.
# 
# To start with, let's look at the 'primary_use' field which appears as a categorical field/label rather numeric values. An active industrial building will ideally consume moe energy than a residential building. This field can be converted to numerical values using the 'LabelEncoder' class. 

# In[ ]:


lbl_encoder = LabelEncoder()
lbl_encoder.fit(train_data_df_m['primary_use'])
train_data_df_m['primary_use'] = np.uint8(lbl_encoder.transform(train_data_df_m['primary_use']))


# In[ ]:


train_data_df_m.head(2)


# The argument continues on what are the fields to be included or excluded as correlating variables in training the machine learning model. A pre-processing technique can be explored to determine what fields to be included or excluded. The 'timestamp' fields has to be transformed or dropped as it's non-numeric. A possible tranfromation among many is to find the log value of the difference in seconds between the 'timestamp' and a reference date e.g. '1970-1-1'.

# In[ ]:


#determine which fields to be dropped
 
for col in ['timestamp','building_id','site_id','meter','meter_type','floor_count','year_built','cloud_coverage','precip_depth_1_hr']:
    del train_data_df_m[col]
 


# In[ ]:


#train_data_df_m = train_data_df_m.dropna() # determine if you really want to drop nan or fill them by interpolation#
train_data_target = train_data_df_m.loc[:,['meter_reading']]
del train_data_df_m['meter_reading']


# In[ ]:


#70% for training and 30% for evaluation/testing
x_train, x_val, y_train, y_val = train_test_split(train_data_df_m,train_data_target, test_size =0.3)


# ### Prediction of Energy Consumption of Buildings
# 
# It will not be clever enough to restrict oneself to a particular. Ideally, different modelling algorithms should be explored and compared. Attracting candidates will be LGBM regressor, support vector regression, random forest regressor etc.
# 
