#!/usr/bin/env python
# coding: utf-8

# # ASHRAE: Automatic FE: Featuretools & Selection FE with SelectFromModel

# **I took my kernel as a basis: [Titanic - Featuretools (automatic FE)](https://www.kaggle.com/vbmokin/titanic-featuretools-automatic-fe)**
# But classifiers were used there, and in our case regressors were needed.

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download datasets](#2)
# 1. [Clearning data and basic FE](#3)
# 1. [Automatic FE with Featuretools](#4)
# 1. [Automatic feature selection (FS)](#5)
#  -  [FS with the Pearson correlation](#5.1)
#  -  [FS with SelectFromModel and LinearSVR](#5.2)
#  -  [FS with SelectFromModel and RandomForestRegressor](#5.3)
# 1. [Comparison of all options of selected features](#6)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to:
# 
# * Automatic FE: https://www.kaggle.com/vbmokin/titanic-featuretools-automatic-fe
# 
# * The main code for basic FE: https://www.kaggle.com/isaienkov/lightgbm-fe-1-19

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import featuretools as ft
from featuretools.primitives import *
from featuretools.variable_types import Numeric

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import LinearSVR
from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.metrics import explained_variance_score
# from sklearn.svm import LinearSVC
# from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE, chi2

# model tuning
# from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

import warnings
warnings.filterwarnings("ignore")


# ## 2. Download datasets <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to https://www.kaggle.com/isaienkov/lightgbm-fe-1-19
building_df = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")
train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")
train = train.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"])
del weather_train
train.head()


# ## 3. Clearning data and basic FE <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


#Thanks to
# basic code: https://www.kaggle.com/isaienkov/lightgbm-fe-1-19
# optimization of format: https://www.kaggle.com/vbmokin/very-significant-safe-memory-lightgbm
train["timestamp"] = pd.to_datetime(train["timestamp"])
train["weekday"] = train["timestamp"].dt.weekday
train["hour"] = train["timestamp"].dt.hour
train["weekday"] = train['weekday'].astype(np.uint8)
train["hour"] = train['hour'].astype(np.uint8)
train['year_built'] = train['year_built']-1900
train['square_feet'] = (10*np.log(train['square_feet'])).astype(np.uint8)

def average_imputation(df, column_name):
    imputation = df.groupby(['timestamp'])[column_name].mean()
    
    df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(lambda x: imputation[df['timestamp'][x.index]].values)
    del imputation
    return df

train = average_imputation(train, 'wind_speed')
train = average_imputation(train, 'wind_direction')

del train["timestamp"]

beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 
          (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

for item in beaufort:
    train.loc[(train['wind_speed']>=item[1]) & (train['wind_speed']<item[2]), 'beaufort_scale'] = item[0]

def degToCompass(num):
    val=int((num/22.5)+.5)
    arr=[i for i in range(0,16)]
    return arr[(val % 16)]

train['wind_direction'] = train['wind_direction'].apply(degToCompass)
train['beaufort_scale'] = train['beaufort_scale'].astype(np.uint8)
train["wind_direction"] = train['wind_direction'].astype(np.uint8)
train["meter"] = train['meter'].astype(np.uint8)
train["site_id"] = train['site_id'].astype(np.uint8)

# Thanks to https://www.kaggle.com/vbmokin/very-significant-safe-memory-lightgbm
train["building_id"] = train['building_id'].astype(np.uint16)
train['air_temperature'] = np.int8(round(2*train['air_temperature'],0)) # store values with precision 0.5
train['cloud_coverage'] = np.uint8(round(10*train['cloud_coverage'],0)) # store values with precision 0.1
train['dew_temperature'] = np.int8(round(5*train['dew_temperature'],0)) # store values with precision 0.2
train['wind_speed'] = np.int8(round(5*train['wind_speed'],0)) # store values with precision 0.2
train['precip_depth_1_hr'] = np.uint8(np.clip(round(train['precip_depth_1_hr'],0),0,255)) # transform [-1,343] to [0,255]
train = train.fillna(0)
train['year_built'] = train['year_built'].astype(np.uint8)
train['floor_count'] = train['floor_count'].astype(np.uint8)

# Thanks to https://www.kaggle.com/isaienkov/lightgbm-fe-1-19
le = LabelEncoder()
train["primary_use"] = le.fit_transform(train["primary_use"])
categoricals = ["site_id", "building_id", "primary_use", "hour", "weekday", "meter",  "wind_direction"]
drop_cols = ["sea_level_pressure", "site_id"]
numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
              "dew_temperature", 'precip_depth_1_hr', 'floor_count', 'beaufort_scale']
feat_cols = categoricals + numericals
target = np.log1p(train["meter_reading"]).astype(np.float16) # float16 - from https://www.kaggle.com/vbmokin/very-significant-safe-memory-lightgbm
train = train.drop(drop_cols, axis = 1)


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.columns


# ### Selection part of data for automatic FE - 10000 meters and it's preprocessing

# In[ ]:


dfe = train.sample(n=10000, replace=True, random_state=1)
dfe = dfe.drop_duplicates(keep=False);
dfe['index'] = dfe.index.tolist()
dfe = dfe.fillna(0); #for cloud and precipitation
dfe.info()


# In[ ]:


#Set all features in Not-negative status
dfe['air_temperature'] = dfe['air_temperature'] - dfe['air_temperature'].min()
dfe['dew_temperature'] = dfe['dew_temperature'] - dfe['dew_temperature'].min()


# In[ ]:


target_fe = dfe['meter_reading']
dfe = dfe.drop(['meter_reading', 'floor_count','year_built'], axis = 1)
del train["meter_reading"]


# ## 4. Automatic FE with Featuretools <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# Thanks to:
# * https://www.kaggle.com/vbmokin/titanic-featuretools-automatic-fe
# * https://www.kaggle.com/liananapalkova/automated-feature-engineering-for-titanic-dataset

# In[ ]:


es = ft.EntitySet(id = 'ashrae_energy_data')
es = es.entity_from_dataframe(entity_id = 'dfe', dataframe = dfe, 
                              variable_types = 
                              {
                                  'air_temperature': ft.variable_types.Numeric,
                                  'cloud_coverage': ft.variable_types.Numeric,
                                  'dew_temperature': ft.variable_types.Numeric,
                                  'precip_depth_1_hr': ft.variable_types.Numeric
                              },
                              index = 'index')


# In[ ]:


es = es.normalize_entity(base_entity_id='dfe', new_entity_id='air_temperature', index='air_temperature')
es = es.normalize_entity(base_entity_id='dfe', new_entity_id='dew_temperature', index='dew_temperature')
es = es.normalize_entity(base_entity_id='dfe', new_entity_id='cloud_coverage', index='cloud_coverage')
es = es.normalize_entity(base_entity_id='dfe', new_entity_id='precip_depth_1_hr', index='precip_depth_1_hr')
es


# In[ ]:


primitives = ft.list_primitives()
primitives[primitives['type'] == 'aggregation'].head(primitives[primitives['type'] == 'aggregation'].shape[0])


# In[ ]:


pd.set_option('max_columns',500)
pd.set_option('max_rows',500)
pd.options.display.max_rows = 300


# In[ ]:


features, feature_names = ft.dfs(entityset = es, 
                                 target_entity = 'dfe', 
                                 max_depth = 2)
features = features.fillna(0)
len(feature_names)


# In[ ]:


features.head()


# In[ ]:


feature_names


# ## 5. Automatic feature selection (FS)<a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


X_norm = MinMaxScaler().fit_transform(train)


# ### 5.1. FS with the Pearson correlation <a class="anchor" id="5.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Absolute value correlation matrix
corr_matrix = features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool));

# Threshold for removing correlated variables
threshold = 0.9

def highlight(value):
    if value > threshold:
        style = 'background-color: pink'
    else:
        style = 'background-color: palegreen'
    return style

# Select columns with correlations above threshold
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]
upper.style.applymap(highlight)


# In[ ]:


features_filtered = features.drop(columns = collinear_features)
features_positive = features_filtered.loc[:, features_filtered.ge(0).all()]
print('The number of features that passed the collinearity threshold: ', features_filtered.shape[1])


# In[ ]:


FE_option1 = features_positive.columns
FE_option1


# ### 5.2. FS with SelectFromModel and LinearSVR <a class="anchor" id="5.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


lsvr = LinearSVR(C=0.05, max_iter = 1000).fit(dfe, target_fe)
model = SelectFromModel(lsvr, prefit=True)
X_new = model.transform(dfe)
X_selected_df = pd.DataFrame(X_new, columns=[dfe.columns[i] for i in range(len(dfe.columns)) if model.get_support()[i]])
X_selected_df.shape


# In[ ]:


FE_option2 = features_positive.columns
FE_option2


# ### 5.3. FS with SelectFromModel and RandomForestRegressor <a class="anchor" id="5.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
embeded_rf_selector = SelectFromModel(model, threshold='1.25*median')
embeded_rf_selector.fit(dfe, target_fe)


# In[ ]:


embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = dfe.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


# In[ ]:


FE_option3 = embeded_rf_feature
FE_option3


# ## 6. Comparison of all options of selected features<a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


FE_all = set(FE_option1).union(set(FE_option2), set(FE_option3))
len(FE_all)


# In[ ]:


FE_all


# In[ ]:


FE_general = set.intersection(set(FE_option1), set(FE_option2), set(FE_option3))
len(FE_all)


# In[ ]:


FE_general

