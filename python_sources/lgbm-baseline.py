#!/usr/bin/env python
# coding: utf-8

# This notebook is a shorter version of a great [EDA and model](https://www.kaggle.com/kulkarnivishwanath/ashrae-great-energy-predictor-iii-eda-model) notebook. It does not provide data overview, just generates the final dataframe and trains a LightGBM classifier. It can be used as a relatively quick baseline for your submission.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

# matplotlib and seaborn for plotting
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

from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import LabelEncoder


# Load the data reducing its size

# In[ ]:


metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}
weather_dtype = {"site_id":"uint8",'air_temperature':"float16",'cloud_coverage':"float16",'dew_temperature':"float16",'precip_depth_1_hr':"float16",
                 'sea_level_pressure':"float32",'wind_direction':"float16",'wind_speed':"float16"}
train_dtype = {'meter':"uint8",'building_id':'uint16'}


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nweather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv", parse_dates=[\'timestamp\'], dtype=weather_dtype)\nweather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv", parse_dates=[\'timestamp\'], dtype=weather_dtype)\n\nmetadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype=metadata_dtype)\n\ntrain = pd.read_csv("../input/ashrae-energy-prediction/train.csv", parse_dates=[\'timestamp\'], dtype=train_dtype)\ntest = pd.read_csv("../input/ashrae-energy-prediction/test.csv", parse_dates=[\'timestamp\'], usecols=[\'building_id\',\'meter\',\'timestamp\'], dtype=train_dtype)\n\nprint(\'Size of train_df data\', train.shape)\nprint(\'Size of weather_train_df data\', weather_train.shape)\nprint(\'Size of weather_test_df data\', weather_test.shape)\nprint(\'Size of building_meta_df data\', metadata.shape)')


# Improve data readability

# In[ ]:


train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)
test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)


# Data overview

# In[ ]:


train.head()


# In[ ]:


weather_train.head()


# In[ ]:


metadata.head()


# In[ ]:


test.head()


# Drop some columns ased on EDA

# In[ ]:


# Dropping floor_count variable as it has 75% missing values
metadata.drop('floor_count',axis=1,inplace=True)


# Construct date features

# In[ ]:


for df in [train, test]:
    df['Month'] = df['timestamp'].dt.month.astype("uint8")
    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")
    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")
    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")


# Convert target to log scale

# In[ ]:


train['meter_reading'] = np.log1p(train['meter_reading'])


# Preprocess metadata 
# 

# In[ ]:


metadata['primary_use'].replace({"Healthcare":"Other","Parking":"Other","Warehouse/storage":"Other","Manufacturing/industrial":"Other",
                                "Retail":"Other","Services":"Other","Technology/science":"Other","Food sales and service":"Other",
                                "Utility":"Other","Religious worship":"Other"},inplace=True)
metadata['square_feet'] = np.log1p(metadata['square_feet'])
metadata['year_built'].fillna(-999, inplace=True)
metadata['year_built'] = metadata['year_built'].astype('int16')


# Merge data

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = pd.merge(train,metadata,on=\'building_id\',how=\'left\')\ntest  = pd.merge(test,metadata,on=\'building_id\',how=\'left\')\nprint ("Training Data+Metadata Shape {}".format(train.shape))\nprint ("Testing Data+Metadata Shape {}".format(test.shape))\ngc.collect()\ntrain = pd.merge(train,weather_train,on=[\'site_id\',\'timestamp\'],how=\'left\')\ntest  = pd.merge(test,weather_test,on=[\'site_id\',\'timestamp\'],how=\'left\')\nprint ("Training Data+Metadata+Weather Shape {}".format(train.shape))\nprint ("Testing Data+Metadata+Weather Shape {}".format(test.shape))\ngc.collect()')


# Prepare data

# In[ ]:


# Save space
for df in [train,test]:
    df['square_feet'] = df['square_feet'].astype('float16')
    
# Fill NA
cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']
for col in cols:
    train[col].fillna(np.nanmean(train[col].tolist()),inplace=True)
    test[col].fillna(np.nanmean(test[col].tolist()),inplace=True)
    
# Drop nonsense entries
# As per the discussion in the following thread, https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117083, there is some discrepancy in the meter_readings for different ste_id's and buildings. It makes sense to delete them
idx_to_drop = list((train[(train['site_id'] == 0) & (train['timestamp'] < "2016-05-21 00:00:00")]).index)
print (len(idx_to_drop))
train.drop(idx_to_drop,axis='rows',inplace=True)

# dropping all the electricity meter readings that are 0, after considering them as anomalies.
idx_to_drop = list(train[(train['meter'] == "Electricity") & (train['meter_reading'] == 0)].index)
print(len(idx_to_drop))
train.drop(idx_to_drop,axis='rows',inplace=True)


# In[ ]:


train.head()


# Measure meter stats

# In[ ]:


get_ipython().run_cell_magic('time', '', "number_unique_meter_per_building = train.groupby('building_id')['meter'].nunique()\ntrain['number_unique_meter_per_building'] = train['building_id'].map(number_unique_meter_per_building)\n\n\nmean_meter_reading_per_building = train.groupby('building_id')['meter_reading'].mean()\ntrain['mean_meter_reading_per_building'] = train['building_id'].map(mean_meter_reading_per_building)\nmedian_meter_reading_per_building = train.groupby('building_id')['meter_reading'].median()\ntrain['median_meter_reading_per_building'] = train['building_id'].map(median_meter_reading_per_building)\nstd_meter_reading_per_building = train.groupby('building_id')['meter_reading'].std()\ntrain['std_meter_reading_per_building'] = train['building_id'].map(std_meter_reading_per_building)\n\n\nmean_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].mean()\ntrain['mean_meter_reading_on_year_built'] = train['year_built'].map(mean_meter_reading_on_year_built)\nmedian_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].median()\ntrain['median_meter_reading_on_year_built'] = train['year_built'].map(median_meter_reading_on_year_built)\nstd_meter_reading_on_year_built = train.groupby('year_built')['meter_reading'].std()\ntrain['std_meter_reading_on_year_built'] = train['year_built'].map(std_meter_reading_on_year_built)\n\n\nmean_meter_reading_per_meter = train.groupby('meter')['meter_reading'].mean()\ntrain['mean_meter_reading_per_meter'] = train['meter'].map(mean_meter_reading_per_meter)\nmedian_meter_reading_per_meter = train.groupby('meter')['meter_reading'].median()\ntrain['median_meter_reading_per_meter'] = train['meter'].map(median_meter_reading_per_meter)\nstd_meter_reading_per_meter = train.groupby('meter')['meter_reading'].std()\ntrain['std_meter_reading_per_meter'] = train['meter'].map(std_meter_reading_per_meter)\n\n\nmean_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].mean()\ntrain['mean_meter_reading_per_primary_usage'] = train['primary_use'].map(mean_meter_reading_per_primary_usage)\nmedian_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].median()\ntrain['median_meter_reading_per_primary_usage'] = train['primary_use'].map(median_meter_reading_per_primary_usage)\nstd_meter_reading_per_primary_usage = train.groupby('primary_use')['meter_reading'].std()\ntrain['std_meter_reading_per_primary_usage'] = train['primary_use'].map(std_meter_reading_per_primary_usage)\n\n\nmean_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].mean()\ntrain['mean_meter_reading_per_site_id'] = train['site_id'].map(mean_meter_reading_per_site_id)\nmedian_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].median()\ntrain['median_meter_reading_per_site_id'] = train['site_id'].map(median_meter_reading_per_site_id)\nstd_meter_reading_per_site_id = train.groupby('site_id')['meter_reading'].std()\ntrain['std_meter_reading_per_site_id'] = train['site_id'].map(std_meter_reading_per_site_id)\n\n\ntest['number_unique_meter_per_building'] = test['building_id'].map(number_unique_meter_per_building)\n\ntest['mean_meter_reading_per_building'] = test['building_id'].map(mean_meter_reading_per_building)\ntest['median_meter_reading_per_building'] = test['building_id'].map(median_meter_reading_per_building)\ntest['std_meter_reading_per_building'] = test['building_id'].map(std_meter_reading_per_building)\n\ntest['mean_meter_reading_on_year_built'] = test['year_built'].map(mean_meter_reading_on_year_built)\ntest['median_meter_reading_on_year_built'] = test['year_built'].map(median_meter_reading_on_year_built)\ntest['std_meter_reading_on_year_built'] = test['year_built'].map(std_meter_reading_on_year_built)\n\ntest['mean_meter_reading_per_meter'] = test['meter'].map(mean_meter_reading_per_meter)\ntest['median_meter_reading_per_meter'] = test['meter'].map(median_meter_reading_per_meter)\ntest['std_meter_reading_per_meter'] = test['meter'].map(std_meter_reading_per_meter)\n\ntest['mean_meter_reading_per_primary_usage'] = test['primary_use'].map(mean_meter_reading_per_primary_usage)\ntest['median_meter_reading_per_primary_usage'] = test['primary_use'].map(median_meter_reading_per_primary_usage)\ntest['std_meter_reading_per_primary_usage'] = test['primary_use'].map(std_meter_reading_per_primary_usage)\n\ntest['mean_meter_reading_per_site_id'] = test['site_id'].map(mean_meter_reading_per_site_id)\ntest['median_meter_reading_per_site_id'] = test['site_id'].map(median_meter_reading_per_site_id)\ntest['std_meter_reading_per_site_id'] = test['site_id'].map(std_meter_reading_per_site_id)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for df in [train, test]:\n    df[\'mean_meter_reading_per_building\'] = df[\'mean_meter_reading_per_building\'].astype("float16")\n    df[\'median_meter_reading_per_building\'] = df[\'mean_meter_reading_per_building\'].astype("float16")\n    df[\'std_meter_reading_per_building\'] = df[\'std_meter_reading_per_building\'].astype("float16")\n    \n    df[\'mean_meter_reading_on_year_built\'] = df[\'mean_meter_reading_on_year_built\'].astype("float16")\n    df[\'median_meter_reading_on_year_built\'] = df[\'median_meter_reading_on_year_built\'].astype("float16")\n    df[\'std_meter_reading_on_year_built\'] = df[\'std_meter_reading_on_year_built\'].astype("float16")\n    \n    df[\'mean_meter_reading_per_meter\'] = df[\'mean_meter_reading_per_meter\'].astype("float16")\n    df[\'median_meter_reading_per_meter\'] = df[\'median_meter_reading_per_meter\'].astype("float16")\n    df[\'std_meter_reading_per_meter\'] = df[\'std_meter_reading_per_meter\'].astype("float16")\n    \n    df[\'mean_meter_reading_per_primary_usage\'] = df[\'mean_meter_reading_per_primary_usage\'].astype("float16")\n    df[\'median_meter_reading_per_primary_usage\'] = df[\'median_meter_reading_per_primary_usage\'].astype("float16")\n    df[\'std_meter_reading_per_primary_usage\'] = df[\'std_meter_reading_per_primary_usage\'].astype("float16")\n    \n    df[\'mean_meter_reading_per_site_id\'] = df[\'mean_meter_reading_per_site_id\'].astype("float16")\n    df[\'median_meter_reading_per_site_id\'] = df[\'median_meter_reading_per_site_id\'].astype("float16")\n    df[\'std_meter_reading_per_site_id\'] = df[\'std_meter_reading_per_site_id\'].astype("float16")\n    \n    df[\'number_unique_meter_per_building\'] = df[\'number_unique_meter_per_building\'].astype(\'uint8\')\ngc.collect()')


# Encode features

# In[ ]:


train.drop('timestamp',axis=1,inplace=True)
test.drop('timestamp',axis=1,inplace=True)

le = LabelEncoder()

train['meter']= le.fit_transform(train['meter']).astype("uint8")
test['meter']= le.fit_transform(test['meter']).astype("uint8")
train['primary_use']= le.fit_transform(train['primary_use']).astype("uint8")
test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")

print (train.shape, test.shape)


# Drop correlated variables

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Let\'s check the correlation between the variables and eliminate the one\'s that have high correlation\n# Threshold for removing correlated variables\nthreshold = 0.9\n\n# Absolute value correlation matrix\ncorr_matrix = train.corr().abs()\n# Upper triangle of correlations\nupper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))\n\n# Select columns with correlations above threshold\nto_drop = [column for column in upper.columns if any(upper[column] > threshold)]\n\nprint(\'There are %d columns to remove.\' % (len(to_drop)))\nprint ("Following columns can be dropped {}".format(to_drop))\n\ntrain.drop(to_drop,axis=1,inplace=True)\ntest.drop(to_drop,axis=1,inplace=True)')


# Split the data for train and validation with stratification by meter reading bins

# In[ ]:


get_ipython().run_cell_magic('time', '', "y = train['meter_reading']\ntrain.drop('meter_reading',axis=1,inplace=True)\ncategorical_cols = ['building_id','Month','meter','Hour','primary_use','DayOfWeek','DayOfMonth']")


# In[ ]:


meter_cut, bins = pd.cut(y, bins=50, retbins=True)
meter_cut.value_counts()


# In[ ]:


# x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.2,random_state=42, stratify=meter_cut)
x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.1,random_state=42)
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)


# In[ ]:


x_train.head()


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


lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_cols)
lgb_test = lgb.Dataset(x_test, y_test, categorical_feature=categorical_cols)
del x_train, x_test , y_train, y_test

params = {'feature_fraction': 0.85, # 0.75
          'bagging_fraction': 0.75,
          'objective': 'regression',
           "num_leaves": 40, # New
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

reg = lgb.train(params, lgb_train, num_boost_round=3000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=100, verbose_eval=100)


# Check feature importance

# In[ ]:


del lgb_train,lgb_test
ser = pd.DataFrame(reg.feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')
ser['Importance'].plot(kind='bar',figsize=(10,6))


# ## Predict

# In[ ]:


del train


# In[ ]:


get_ipython().run_cell_magic('time', '', 'predictions = []\nstep = 50000\nfor i in range(0, len(test), step):\n    predictions.extend(np.expm1(reg.predict(test.iloc[i: min(i+step, len(test)), :], num_iteration=reg.best_iteration)))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'Submission = pd.DataFrame(test.index,columns=[\'row_id\'])\nSubmission[\'meter_reading\'] = predictions\nSubmission[\'meter_reading\'].clip(lower=0,upper=None,inplace=True)\nSubmission.to_csv("lgbm_fill_na.csv",index=None)')


# In[ ]:




