#!/usr/bin/env python
# coding: utf-8

# Team: RF  
# Members: Jose Rodrigo Flores Espinosa  
# 
# This kernel addreses the challenge posed by ASHRAE and titled: "ASHRAE - Great Energy Predictor III - How much energy will a building consume?"
# Details about the problem posed can be found in the [main page of the competition](https://www.kaggle.com/c/ashrae-energy-prediction).
# 
# 

# Imports

# In[ ]:


import numpy as np
import pandas as pd 
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
pd.set_option('max_columns', 150)

py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import os
import random
import math
import psutil
import pickle
import timeit

from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import LabelEncoder

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas_profiling


# Load the data reducing its size

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmetadata_dtype = {\'site_id\':"uint8",\'building_id\':\'uint16\',\'square_feet\':\'float32\',\'year_built\':\'float32\',\'floor_count\':"float16"}\nweather_dtype = {"site_id":"uint8",\'air_temperature\':"float16",\'cloud_coverage\':"float16",\'dew_temperature\':"float16",\'precip_depth_1_hr\':"float16",\n                 \'sea_level_pressure\':"float32",\'wind_direction\':"float16",\'wind_speed\':"float16"}\ntrain_dtype = {\'meter\':"uint8",\'building_id\':\'uint16\',\'meter_reading\':"float32"}\n\nstart_time = timeit.default_timer()\n\nweather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv", parse_dates=[\'timestamp\'], dtype=weather_dtype)\n# weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv", parse_dates=[\'timestamp\'], dtype=weather_dtype)\nmetadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype=metadata_dtype)\ntrain = pd.read_csv("../input/ashrae-energy-prediction/train.csv", parse_dates=[\'timestamp\'], dtype=train_dtype)\n# test = pd.read_csv("../input/ashrae-energy-prediction/test.csv", parse_dates=[\'timestamp\'], usecols=[\'building_id\',\'meter\',\'timestamp\'], dtype=train_dtype)\n\nprint(\'Size of train_df data\', train.shape)\nprint(\'Size of weather_train_df data\', weather_train.shape)\n# print(\'Size of weather_test_df data\', weather_test.shape)\nprint(\'Size of building_meta_df data\', metadata.shape)\n\nelapsed = timeit.default_timer() - start_time\nprint(elapsed)')


# In[ ]:


weather_train.head()
metadata.head()
train.head()


# Data Overview 

# In[ ]:


# start_time = timeit.default_timer()

# missing_weather = pd.DataFrame(weather_train.isna().sum()/len(weather_train),columns=["Weather_Train_Missing_Pct"])
# # missing_weather["Weather_Test_Missing_Pct"] = weather_test.isna().sum()/len(weather_test)
# missing_weather

# missing_metadata = pd.DataFrame(metadata.isna().sum()/len(metadata),columns=["Metadada_Missing"])
# missing_metadata

# weather_train_report = weather_train.profile_report(style={'full_width':True},title='Weather Data Profiling Report')
# weather_train_report

# metadata_report = metadata.profile_report(style={'full_width':True},title='Metadata Profiling Report')
# metadata_report

# elapsed = timeit.default_timer() - start_time
# print(elapsed)

# # In[5]:

# weather_train.head()
# metadata.head()
# train.head()

# del missing_weather
# del missing_metadata
# del weather_train_report
# del metadata_report
# gc.collect()


# In[ ]:


# cols = list(weather_train.columns[2:])
# cols_imputed = weather_train[cols].isnull().astype('bool_').add_suffix('_imputed')

# imp = IterativeImputer(max_iter=10, verbose=0)
# imp.fit(weather_train.iloc[:,2:])
# weather_train_imputed = imp.transform(weather_train.iloc[:,2:])
# weather_train_imputed = pd.concat([weather_train.iloc[:,0:2],pd.DataFrame(weather_train_imputed, columns=weather_train.columns[2:]), cols_imputed], axis=1)
# # weather_train_imputed = pd.concat([weather_train.iloc[:,0:2],pd.DataFrame(weather_train_imputed, columns=weather_train.columns[2:])], axis=1)
# pd.DataFrame(weather_train_imputed.isna().sum()/len(weather_train_imputed),columns=["Weather_Train_Missing_Imputed"])

# # cols = list(weather_test.columns[2:])
# # cols_imputed = weather_test[cols].isnull().astype('bool_').add_suffix('_imputed')

# # imp = IterativeImputer(max_iter=10, verbose=0)
# # imp.fit(weather_test.iloc[:,2:])
# # weather_test_imputed = imp.transform(weather_test.iloc[:,2:])
# # weather_test_imputed = pd.concat([weather_test.iloc[:,0:2],pd.DataFrame(weather_test_imputed, columns=weather_test.columns[2:]), cols_imputed], axis=1)
# # weather_test_imputed = pd.concat([weather_test.iloc[:,0:2],pd.DataFrame(weather_test_imputed, columns=weather_test.columns[2:])], axis=1)
# # pd.DataFrame(weather_test_imputed.isna().sum()/len(weather_test_imputed),columns=["Weather_Train_Missing_Imputed"])

# # imputation floor_count & year built

# cols = list(metadata.columns[4:])
# cols_imputed = metadata[cols].isnull().astype('uint8').add_suffix('_imputed')

# imp = IterativeImputer(max_iter=10, verbose=0)
# imp.fit(metadata.iloc[:,3:])
# metadata_imputed = imp.transform(metadata.iloc[:,3:])
# metadata_imputed = pd.concat([metadata.iloc[:,0:3],pd.DataFrame(metadata_imputed, columns=metadata.columns[3:]), cols_imputed], axis=1)
# #metadata_imputed = pd.concat([metadata.iloc[:,0:3],pd.DataFrame(metadata_imputed, columns=metadata.columns[3:])], axis=1)
# pd.DataFrame(metadata_imputed.isna().sum()/len(metadata_imputed),columns=["Metadata_Missing_Imputed"])

# metadata_imputed.year_built = metadata_imputed.year_built.round()
# metadata_imputed.floor_count = metadata_imputed.floor_count.round()

# del weather_train
# # del weather_test
# del metadata
# del cols
# del cols_imputed

# gc.collect()


# Construct date features

# In[ ]:


# for df in [train, test]:
for df in [train]:
    df['Month'] = df['timestamp'].dt.month.astype("uint8")
    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")
    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")
    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")
    df['timestamp_2'] = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    df['timestamp_2'] = df.timestamp_2.astype('uint16')
    
# Code to read and combine the standard input files, converting timestamps to number of hours since the beginning of 2016.

# weather_train_imputed['timestamp_2'] = (weather_train_imputed.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
# weather_train_imputed['timestamp_2'] = weather_train_imputed.timestamp_2.astype('int16')

weather_train['timestamp_2'] = (weather_train.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
weather_train['timestamp_2'] = weather_train.timestamp_2.astype('int16')

# weather_test_imputed['timestamp_2'] = (weather_test_imputed.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
# weather_test_imputed['timestamp_2'] = weather_test_imputed.timestamp_2.astype('int16')

#fix timestamps

site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]
GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}
weather_train.timestamp_2 = weather_train.timestamp_2 + weather_train.site_id.map(GMT_offset_map)
# weather_test_imputed.timestamp_2 = weather_test_imputed.timestamp_2 + weather_test_imputed.site_id.map(GMT_offset_map)

weather_train.drop('timestamp',axis=1,inplace=True)
# weather_test_imputed.drop('timestamp',axis=1,inplace=True)
gc.collect()


# Data treatment
# 
# .- Drop some columns ased on EDA  
# .- Convert target to log scale  
# .- Preprocess metadata  
# 
# 
# 

# In[ ]:


# Dropping floor_count variable as it has 75% missing values
# metadata_imputed.drop('floor_count',axis=1,inplace=True)

train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)
#test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

train.rename(columns={'timestamp':'timestamp_train'}, inplace=True)
# test.rename(columns={'timestamp':'timestamp_test'}, inplace=True)

train['meter_reading'] = np.log1p(train['meter_reading'])

metadata['primary_use'].replace({"Healthcare":"Other","Parking":"Other","Warehouse/storage":"Other","Manufacturing/industrial":"Other",
                                "Retail":"Other","Services":"Other","Technology/science":"Other","Food sales and service":"Other",
                                "Utility":"Other","Religious worship":"Other"},inplace=True)
metadata['square_feet'] = np.log1p(metadata['square_feet'])
# metadata_imputed['year_built'].fillna(-999, inplace=True)
# metadata_imputed['square_feet'] = metadata_imputed['square_feet'].astype('float16')
# metadata_imputed['year_built'] = metadata_imputed['year_built'].astype('uint16')
# metadata_imputed['floor_count'] = metadata_imputed['floor_count'].astype('uint8')

gc.collect()


# Merge data

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = pd.merge(train,metadata,on=\'building_id\',how=\'left\')\n# test  = pd.merge(test,metadata_imputed,on=\'building_id\',how=\'left\')\nprint ("Training Data+Metadata Shape {}".format(train.shape))\n# print ("Testing Data+Metadata Shape {}".format(test.shape))\ndel metadata\ngc.collect()\n\ntrain = pd.merge(train,weather_train,on=[\'site_id\',\'timestamp_2\'],how=\'left\')\ndel weather_train\ngc.collect()\n\n# test  = pd.merge(test,weather_test_imputed,on=[\'site_id\',\'timestamp_2\'],how=\'left\')\nprint ("Training Data+Metadata+Weather Shape {}".format(train.shape))\n# print ("Testing Data+Metadata+Weather Shape {}".format(test.shape))\n\n# del weather_test_imputed\n# gc.collect()\n\n#missing_train = pd.DataFrame(train.isna().sum()/len(train),columns=["Train_Missing"])\n#missing_train\n\n#missing_test = pd.DataFrame(test.isna().sum()/len(test),columns=["Train_Missing"])\n#missing_test\n')


# Prepare data

# In[ ]:


# Save space
# commented since already done above
#for df in [train,test]:
#    df['square_feet'] = df['square_feet'].astype('float16')
    
# Fill NA

#cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']
#for col in cols:
#    train[col].fillna(train[col].mean(),inplace=True)
#    test[col].fillna(test[col].mean(),inplace=True)
    
# Drop nonsense entries
# As per the discussion in the following thread, https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117083, there is some discrepancy in the meter_readings for different ste_id's and buildings. It makes sense to delete them
idx_to_drop = list((train[(train['site_id'] == 0) & (train['timestamp_train'] < "2016-05-21 00:00:00")]).index)
print (len(idx_to_drop))
train.drop(idx_to_drop,axis='rows',inplace=True)

# dropping all the electricity meter readings that are 0, after considering them as anomalies.
idx_to_drop = list(train[(train['meter'] == "Electricity") & (train['meter_reading'] == 0)].index)
# idx_to_drop = list(train[(train['meter'] == 0) & (train['meter_reading'] == 0)].index)
print(len(idx_to_drop))
train.drop(idx_to_drop,axis='rows',inplace=True)

##train.drop('timestamp',axis=1,inplace=True)
##test.drop('timestamp',axis=1,inplace=True)

train.drop('timestamp_train',axis=1,inplace=True)
# test.drop('timestamp_test',axis=1,inplace=True)

train.drop('timestamp_2',axis=1,inplace=True)
train = train.reset_index()
train.drop('index',axis=1,inplace=True)

# test.drop('timestamp_2',axis=1,inplace=True)

del idx_to_drop
gc.collect()

# Encode features
le = LabelEncoder()

train['meter']= le.fit_transform(train['meter']).astype("uint8")
# test['meter']= le.fit_transform(test['meter']).astype("uint8")
train['primary_use']= le.fit_transform(train['primary_use']).astype("uint8")
# test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")

# print (train.shape, test.shape)
print (train.shape)


# #### Imputation

# In[ ]:


cols = list(train.columns[10:])
cols_imputed = train[cols].isnull().astype('uint8').add_suffix('_imputed')

imp = IterativeImputer(max_iter=10, verbose=0)
imp.fit(train.iloc[:,8:])
train_temp = imp.transform(train.iloc[:,8:])
train = pd.concat([train.iloc[:,0:8],pd.DataFrame(train_temp, columns=train.columns[8:]), cols_imputed], axis=1)
# weather_train_imputed = pd.concat([weather_train.iloc[:,0:2],pd.DataFrame(weather_train_imputed, columns=weather_train.columns[2:])], axis=1)
pd.DataFrame(train.isna().sum()/len(train),columns=["Weather_Train_Missing_Imputed"])

del train_temp
del cols
del cols_imputed
gc.collect()

train['square_feet'] = train['square_feet'].astype('float16')
train['year_built'] = train['year_built'].astype('uint16')
train['floor_count'] = train['floor_count'].astype('uint8')
train['primary_use'] = train['primary_use'].astype('uint8')
train['air_temperature'] = train['air_temperature'].astype('float16')
train['cloud_coverage'] = train['cloud_coverage'].astype('float16')
train['dew_temperature'] = train['dew_temperature'].astype('float16')
train['precip_depth_1_hr'] = train['precip_depth_1_hr'].astype('float16')
train['wind_direction'] = train['wind_direction'].astype('float16')
train['wind_speed'] = train['wind_speed'].astype('float16')


# Measure meter stats

# In[ ]:


get_ipython().run_cell_magic('time', '', "number_unique_meter_per_building = train.groupby('building_id')['meter'].nunique()\ntrain['number_unique_meter_per_building'] = train['building_id'].map(number_unique_meter_per_building)\nmean_meter_reading_per_building = train.groupby('building_id')['meter_reading'].mean()\ntrain['mean_meter_reading_per_building'] = train['building_id'].map(mean_meter_reading_per_building)\n# median_meter_reading_per_building = train.groupby('building_id')['meter_reading'].median()\n# train['median_meter_reading_per_building'] = train['building_id'].map(median_meter_reading_per_building)\nstd_meter_reading_per_building = train.groupby('building_id')['meter_reading'].std()\ntrain['std_meter_reading_per_building'] = train['building_id'].map(std_meter_reading_per_building)\nmean_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].mean()\ntrain['mean_meter_reading_on_year_built'] = train['year_built'].map(mean_meter_reading_on_year_built)\n# median_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].median()\n# train['median_meter_reading_on_year_built'] = train['year_built'].map(median_meter_reading_on_year_built)\nstd_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].std()\ntrain['std_meter_reading_on_year_built'] = train['year_built'].map(std_meter_reading_on_year_built)\nmean_meter_reading_per_meter = train.groupby('meter')['meter_reading'].mean()\ntrain['mean_meter_reading_per_meter'] = train['meter'].map(mean_meter_reading_per_meter)\n# median_meter_reading_per_meter = train.groupby('meter')['meter_reading'].median()\n# train['median_meter_reading_per_meter'] = train['meter'].map(median_meter_reading_per_meter)\nstd_meter_reading_per_meter = train.groupby('meter')['meter_reading'].std()\ntrain['std_meter_reading_per_meter'] = train['meter'].map(std_meter_reading_per_meter)\nmean_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].mean()\ntrain['mean_meter_reading_per_primary_usage'] = train['primary_use'].map(mean_meter_reading_per_primary_usage)\n# median_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].median()\n# train['median_meter_reading_per_primary_usage'] = train['primary_use'].map(median_meter_reading_per_primary_usage)\nstd_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].std()\ntrain['std_meter_reading_per_primary_usage'] = train['primary_use'].map(std_meter_reading_per_primary_usage)\nmean_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].mean()\ntrain['mean_meter_reading_per_site_id'] = train['site_id'].map(mean_meter_reading_per_site_id)\n# median_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].median()\n# train['median_meter_reading_per_site_id'] = train['site_id'].map(median_meter_reading_per_site_id)\nstd_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].std()\ntrain['std_meter_reading_per_site_id'] = train['site_id'].map(std_meter_reading_per_site_id)\n\n\n# test['number_unique_meter_per_building'] = test['building_id'].map(number_unique_meter_per_building)\n\n# test['mean_meter_reading_per_building'] = test['building_id'].map(mean_meter_reading_per_building)\n# test['median_meter_reading_per_building'] = test['building_id'].map(median_meter_reading_per_building)\n# test['std_meter_reading_per_building'] = test['building_id'].map(std_meter_reading_per_building)\n\n# test['mean_meter_reading_on_year_built'] = test['year_built'].map(mean_meter_reading_on_year_built)\n# test['median_meter_reading_on_year_built'] = test['year_built'].map(median_meter_reading_on_year_built)\n# test['std_meter_reading_on_year_built'] = test['year_built'].map(std_meter_reading_on_year_built)\n\n# test['mean_meter_reading_per_meter'] = test['meter'].map(mean_meter_reading_per_meter)\n# test['median_meter_reading_per_meter'] = test['meter'].map(median_meter_reading_per_meter)\n# test['std_meter_reading_per_meter'] = test['meter'].map(std_meter_reading_per_meter)\n\n# test['mean_meter_reading_per_primary_usage'] = test['primary_use'].map(mean_meter_reading_per_primary_usage)\n# test['median_meter_reading_per_primary_usage'] = test['primary_use'].map(median_meter_reading_per_primary_usage)\n# test['std_meter_reading_per_primary_usage'] = test['primary_use'].map(std_meter_reading_per_primary_usage)\n\n# test['mean_meter_reading_per_site_id'] = test['site_id'].map(mean_meter_reading_per_site_id)\n# test['median_meter_reading_per_site_id'] = test['site_id'].map(median_meter_reading_per_site_id)\n# test['std_meter_reading_per_site_id'] = test['site_id'].map(std_meter_reading_per_site_id)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# for df in [train, test]:\nfor df in [train]:\n    df[\'mean_meter_reading_per_building\'] = df[\'mean_meter_reading_per_building\'].astype("float16")\n#     df[\'median_meter_reading_per_building\'] = df[\'mean_meter_reading_per_building\'].astype("float16")\n    df[\'std_meter_reading_per_building\'] = df[\'std_meter_reading_per_building\'].astype("float16")\n    df[\'mean_meter_reading_on_year_built\'] = df[\'mean_meter_reading_on_year_built\'].astype("float16")\n#     df[\'median_meter_reading_on_year_built\'] = df[\'median_meter_reading_on_year_built\'].astype("float16")\n    df[\'std_meter_reading_on_year_built\'] = df[\'std_meter_reading_on_year_built\'].astype("float16")\n    df[\'mean_meter_reading_per_meter\'] = df[\'mean_meter_reading_per_meter\'].astype("float16")\n#     df[\'median_meter_reading_per_meter\'] = df[\'median_meter_reading_per_meter\'].astype("float16")\n    df[\'std_meter_reading_per_meter\'] = df[\'std_meter_reading_per_meter\'].astype("float16")\n    df[\'mean_meter_reading_per_primary_usage\'] = df[\'mean_meter_reading_per_primary_usage\'].astype("float16")\n#     df[\'median_meter_reading_per_primary_usage\'] = df[\'median_meter_reading_per_primary_usage\'].astype("float16")\n    df[\'std_meter_reading_per_primary_usage\'] = df[\'std_meter_reading_per_primary_usage\'].astype("float16")\n    df[\'mean_meter_reading_per_site_id\'] = df[\'mean_meter_reading_per_site_id\'].astype("float16")\n#     df[\'median_meter_reading_per_site_id\'] = df[\'median_meter_reading_per_site_id\'].astype("float16")\n    df[\'std_meter_reading_per_site_id\'] = df[\'std_meter_reading_per_site_id\'].astype("float16")\n    df[\'number_unique_meter_per_building\'] = df[\'number_unique_meter_per_building\'].astype(\'uint8\')\n\ngc.collect()\n\n# Following columns can be dropped \n# [\'site_id\', \'median_meter_reading_per_building\', \'median_meter_reading_on_year_built\', \'median_meter_reading_per_meter\', \n# \'median_meter_reading_per_primary_usage\', \'median_meter_reading_per_site_id\']')


# Drop correlated variables

# In[ ]:


# %%time
# # Let's check the correlation between the variables and eliminate the one's that have high correlation
# # Threshold for removing correlated variables
# threshold = 0.9

# # Absolute value correlation matrix
# corr_matrix = train.corr().abs()
# # Upper triangle of correlations
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# del corr_matrix
# gc.collect()

# # Select columns with correlations above threshold
# to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
# del upper 
# gc.collect()

# print('There are %d columns to remove.' % (len(to_drop)))
# print ("Following columns can be dropped {}".format(to_drop))

# train.drop(to_drop, axis=1, inplace=True)
# # test.drop(to_drop,axis=1,inplace=True)
# # del to_drop
# gc.collect()


# Split the data for train and validation with stratification by meter reading bins

# In[ ]:


get_ipython().run_cell_magic('time', '', "y = train['meter_reading']\ntrain.drop('meter_reading',axis=1,inplace=True)\ntrain.drop('site_id',axis=1,inplace=True)\ntrain.drop('floor_count_imputed',axis=1,inplace=True)\n\ncategorical_cols = ['building_id','meter','Month','DayOfMonth','DayOfWeek','Hour','primary_use',\n                    'year_built','floor_count','year_built_imputed','air_temperature_imputed',\n                    'cloud_coverage_imputed','precip_depth_1_hr_imputed','sea_level_pressure_imputed','wind_direction_imputed',\n                    'wind_speed_imputed']\n")


# In[ ]:


meter_cut, bins = pd.cut(y, bins=50, retbins=True)
meter_cut.value_counts()


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.2,random_state=42, stratify=meter_cut)
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)

train_columns = train.columns
del train
del meter_cut
del bins
gc.collect()


# Make dummies if necessary -- for RF

# x_train = pd.get_dummies(x_train, columns=categorical_cols, sparse=True)
# 
# x_test = pd.get_dummies(x_test, columns=categorical_cols, sparse=True)
# 
# gc.collect()
# 
# x_train.shape

# ## Model
# 
# Train baseline model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor as RF
import lightgbm as lgb


# In[ ]:


lgb_train = lgb.Dataset(x_train, y_train,categorical_feature=categorical_cols)
lgb_test = lgb.Dataset(x_test, y_test,categorical_feature=categorical_cols)
del x_train, x_test , y_train, y_test
gc.collect()

params = {'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.15,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'rmse',
          "verbosity": -1,
          'reg_alpha': 0.5,
          'reg_lambda': 0.5,
          'random_state': 47
         }

reg = lgb.train(params, lgb_train, num_boost_round=3000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=100, verbose_eval = 100)

del lgb_train, lgb_test
gc.collect() 


# Check feature importance

# In[ ]:


# ser = pd.DataFrame(reg.feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')
ser = pd.DataFrame(reg.feature_importance(),train_columns,columns=['Importance']).sort_values(by='Importance')
ser['Importance'].plot(kind='bar',figsize=(10,6))

#del train
del ser
del train_columns
gc.collect() 


# #### loading and processing of test objects

# In[ ]:


test = pd.read_csv("../input/ashrae-energy-prediction/test.csv", parse_dates=['timestamp'], usecols=['building_id','meter','timestamp'], dtype=train_dtype)

#
for df in [test]:
    df['Month'] = df['timestamp'].dt.month.astype("uint8")
    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")
    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")
    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")
    df['timestamp_2'] = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    df['timestamp_2'] = df.timestamp_2.astype('uint16')

site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]
GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}

# weather_test_imputed.timestamp_2 = weather_test_imputed.timestamp_2 + weather_test_imputed.site_id.map(GMT_offset_map)
# weather_test_imputed.drop('timestamp',axis=1,inplace=True)
# weather_test.timestamp_2 = weather_test.timestamp_2 + weather_test.site_id.map(GMT_offset_map)
# weather_test.drop('timestamp',axis=1,inplace=True)

#
test.rename(columns={'timestamp':'timestamp_test'}, inplace=True)

#
##########
metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype=metadata_dtype)
# cols = list(metadata.columns[4:])
# cols_imputed = metadata[cols].isnull().astype('uint8').add_suffix('_imputed')

# imp = IterativeImputer(max_iter=10, verbose=0)
# imp.fit(metadata.iloc[:,3:])
# metadata_imputed = imp.transform(metadata.iloc[:,3:])
# metadata_imputed = pd.concat([metadata.iloc[:,0:3],pd.DataFrame(metadata_imputed, columns=metadata.columns[3:]), cols_imputed], axis=1)
# #metadata_imputed = pd.concat([metadata.iloc[:,0:3],pd.DataFrame(metadata_imputed, columns=metadata.columns[3:])], axis=1)
# pd.DataFrame(metadata_imputed.isna().sum()/len(metadata_imputed),columns=["Metadata_Missing_Imputed"])

# metadata_imputed.year_built = metadata_imputed.year_built.round()
# metadata_imputed.floor_count = metadata_imputed.floor_count.round()

# del metadata
# del cols
# del cols_imputed
# gc.collect()

##########

test  = pd.merge(test,metadata,on='building_id',how='left')
print ("Testing Data+Metadata Shape {}".format(test.shape))
del metadata
gc.collect()

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv", parse_dates=['timestamp'], dtype=weather_dtype)
print('Size of weather_test_df data', weather_test.shape)

weather_test['timestamp_2'] = (weather_test.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
weather_test['timestamp_2'] = weather_test.timestamp_2.astype('int16')

weather_test.timestamp_2 = weather_test.timestamp_2 + weather_test.site_id.map(GMT_offset_map)
weather_test.drop('timestamp',axis=1,inplace=True)

test  = pd.merge(test,weather_test,on=['site_id','timestamp_2'],how='left')
print ("Testing Data+Metadata+Weather Shape {}".format(test.shape))

test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)
del weather_test
gc.collect()

#
test.drop('timestamp_test',axis=1,inplace=True)
test.drop('timestamp_2',axis=1,inplace=True)
gc.collect()

le = LabelEncoder()
test['meter']= le.fit_transform(test['meter']).astype("uint8")
test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")
print(test.shape)

#


# In[ ]:


################################################
# Imputation

cols = list(test.columns[9:])
cols_imputed = test[cols].isnull().astype('uint8').add_suffix('_imputed')

imp = IterativeImputer(max_iter=10, verbose=0)
imp.fit(test.iloc[:,7:])
test_temp = imp.transform(test.iloc[:,7:])
test = pd.concat([test.iloc[:,0:7],pd.DataFrame(test_temp, columns=test.columns[7:]), cols_imputed], axis=1)
test.drop('site_id',axis=1,inplace=True)
test.drop('floor_count_imputed',axis=1,inplace=True)
# weather_train_imputed = pd.concat([weather_train.iloc[:,0:2],pd.DataFrame(weather_train_imputed, columns=weather_train.columns[2:])], axis=1)
pd.DataFrame(test.isna().sum()/len(test),columns=["Weather_Train_Missing_Imputed"])

del test_temp
del cols
del cols_imputed
gc.collect()

test['square_feet'] = test['square_feet'].astype('float16')
test['year_built'] = test['year_built'].astype('uint16')
test['floor_count'] = test['floor_count'].astype('uint8')
test['primary_use'] = test['primary_use'].astype('uint8')
test['air_temperature'] = test['air_temperature'].astype('float16')
test['cloud_coverage'] = test['cloud_coverage'].astype('float16')
test['dew_temperature'] = test['dew_temperature'].astype('float16')
test['precip_depth_1_hr'] = test['precip_depth_1_hr'].astype('float16')
test['wind_direction'] = test['wind_direction'].astype('float16')
test['wind_speed'] = test['wind_speed'].astype('float16')



################################################


#
test['number_unique_meter_per_building'] = test['building_id'].map(number_unique_meter_per_building)
test['number_unique_meter_per_building'] = test['number_unique_meter_per_building'].astype('uint8')
test['mean_meter_reading_per_building'] = test['building_id'].map(mean_meter_reading_per_building)
test['mean_meter_reading_per_building'] = test['mean_meter_reading_per_building'].astype("float16")
# test['median_meter_reading_per_building'] = test['building_id'].map(median_meter_reading_per_building)
test['std_meter_reading_per_building'] = test['building_id'].map(std_meter_reading_per_building)
test['std_meter_reading_per_building'] = test['std_meter_reading_per_building'].astype("float16")
test['mean_meter_reading_on_year_built'] = test['year_built'].map(mean_meter_reading_on_year_built)
test['mean_meter_reading_on_year_built'] = test['mean_meter_reading_on_year_built'].astype("float16")
# test['median_meter_reading_on_year_built'] = test['year_built'].map(median_meter_reading_on_year_built)
test['std_meter_reading_on_year_built'] = test['year_built'].map(std_meter_reading_on_year_built)
test['std_meter_reading_on_year_built'] = test['std_meter_reading_on_year_built'].astype("float16")
test['mean_meter_reading_per_meter'] = test['meter'].map(mean_meter_reading_per_meter)
test['mean_meter_reading_per_meter'] = test['mean_meter_reading_per_meter'].astype("float16")
# test['median_meter_reading_per_meter'] = test['meter'].map(median_meter_reading_per_meter)
test['std_meter_reading_per_meter'] = test['meter'].map(std_meter_reading_per_meter)
test['std_meter_reading_per_meter'] = test['std_meter_reading_per_meter'].astype("float16")
test['mean_meter_reading_per_primary_usage'] = test['primary_use'].map(mean_meter_reading_per_primary_usage)
test['mean_meter_reading_per_primary_usage'] = test['mean_meter_reading_per_primary_usage'].astype("float16")
# test['median_meter_reading_per_primary_usage'] = test['primary_use'].map(median_meter_reading_per_primary_usage)
test['std_meter_reading_per_primary_usage'] = test['primary_use'].map(std_meter_reading_per_primary_usage)
test['std_meter_reading_per_primary_usage'] = test['std_meter_reading_per_primary_usage'].astype("float16")
test['mean_meter_reading_per_site_id'] = test['site_id'].map(mean_meter_reading_per_site_id)
test['mean_meter_reading_per_site_id'] = test['mean_meter_reading_per_site_id'].astype("float16")
# test['median_meter_reading_per_site_id'] = test['site_id'].map(median_meter_reading_per_site_id)
test['std_meter_reading_per_site_id'] = test['site_id'].map(std_meter_reading_per_site_id)
test['std_meter_reading_per_site_id'] = test['std_meter_reading_per_site_id'].astype("float16")

#
# for df in [test]:
#     df['mean_meter_reading_per_building'] = df['mean_meter_reading_per_building'].astype("float16")
#     df['median_meter_reading_per_building'] = df['mean_meter_reading_per_building'].astype("float16")
#     df['std_meter_reading_per_building'] = df['std_meter_reading_per_building'].astype("float16")
#     df['mean_meter_reading_on_year_built'] = df['mean_meter_reading_on_year_built'].astype("float16")
#     df['median_meter_reading_on_year_built'] = df['median_meter_reading_on_year_built'].astype("float16")
#     df['std_meter_reading_on_year_built'] = df['std_meter_reading_on_year_built'].astype("float16")
#     df['mean_meter_reading_per_meter'] = df['mean_meter_reading_per_meter'].astype("float16")
#     df['median_meter_reading_per_meter'] = df['median_meter_reading_per_meter'].astype("float16")
#     df['std_meter_reading_per_meter'] = df['std_meter_reading_per_meter'].astype("float16")
#     df['mean_meter_reading_per_primary_usage'] = df['mean_meter_reading_per_primary_usage'].astype("float16")
#     df['median_meter_reading_per_primary_usage'] = df['median_meter_reading_per_primary_usage'].astype("float16")
#     df['std_meter_reading_per_primary_usage'] = df['std_meter_reading_per_primary_usage'].astype("float16")
#     df['mean_meter_reading_per_site_id'] = df['mean_meter_reading_per_site_id'].astype("float16")
#     df['median_meter_reading_per_site_id'] = df['median_meter_reading_per_site_id'].astype("float16")
#     df['std_meter_reading_per_site_id'] = df['std_meter_reading_per_site_id'].astype("float16")
#     df['number_unique_meter_per_building'] = df['number_unique_meter_per_building'].astype('uint8')


#
# le = LabelEncoder()
# test['meter']= le.fit_transform(test['meter']).astype("uint8")
# test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")
# print(test.shape)

# #
# test.drop(to_drop,axis=1,inplace=True)
# del to_drop
gc.collect()


# ## Predict

# In[ ]:


get_ipython().run_cell_magic('time', '', 'predictions = []\nstep = 50000\nfor i in range(0, len(test), step):\n    predictions.extend(np.expm1(reg.predict(test.iloc[i: min(i+step, len(test)), :], num_iteration=reg.best_iteration)))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'Submission = pd.DataFrame(test.index,columns=[\'row_id\'])\nSubmission[\'meter_reading\'] = predictions\nSubmission[\'meter_reading\'].clip(lower=0,upper=None,inplace=True)\nSubmission.to_csv("lgbm.csv",index=None)')

