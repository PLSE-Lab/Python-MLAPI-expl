#!/usr/bin/env python
# coding: utf-8

# # Purpose of this notebook

# * **The purpose of this notebook is to handle missing data present in building metadata, weather train files**
# * **This notebook outputs cleaned data files with names cleaned_building_metadata, cleaned_weather_train respectively. **
# * **The output of this notebook can be used to create new notebook with these cleaned files to perform furthur steps like model fitting etc.**
# * **Steps to use these cleaned files: Output->New_notebook**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from datetime import datetime
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Analysis for building_metadata

# In[ ]:


building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')


# In[ ]:


building_metadata.shape


# In[ ]:


building_metadata.head()


# In[ ]:


building_metadata.shape


# In[ ]:


building_metadata['primary_use'].isna().sum()


# In[ ]:


building_metadata['square_feet'].isna().sum()


# In[ ]:


building_metadata['year_built'].isna().sum()


# In[ ]:


building_metadata['floor_count'].isna().sum()


# Deleting this column as it has too many null values

# In[ ]:


del building_metadata['floor_count']


# In[ ]:


building_metadata.to_csv('cleaned_building_metadata.csv',index=False)


# In[ ]:


del building_metadata
gc.collect()


# # Handling missing data of table weather data (train)

# In[ ]:


weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')


# In[ ]:


weather_train.shape


# In[ ]:


weather_train.head()


# In[ ]:


weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])


# In[ ]:


weather_train['date'] = weather_train['timestamp'].apply(datetime.date)


# In[ ]:


weather_train['time'] = weather_train['timestamp'].apply(datetime.time)


# **Handling missing air_temperature column**

# In[ ]:


weather_train['air_temperature'].isna().sum()


# In[ ]:


temp = weather_train.groupby(['site_id','date','time'])['air_temperature'].agg(pd.Series.mode)


# In[ ]:


temp


# In[ ]:


temp_list = []
for i in weather_train.index:
    if np.isnan(weather_train.loc[i].air_temperature):
        try:
            temp1 = temp[[weather_train.loc[i].site_id,weather_train.loc[i].date,weather_train.loc[i].time]]
            temp_list.append(temp1)
        except:
            temp_list.append(np.nan)
    else:
        temp_list.append(weather_train.loc[i].air_temperature)


# In[ ]:


weather_train['air_temperature'] = temp_list


# In[ ]:


temp1 = []
for i in weather_train.index:
    if type(weather_train.loc[i].air_temperature) == np.float64:
        temp = weather_train.loc[i].air_temperature
        temp1.append(temp)
    else:
        temp2 = 0
        temp3 = 0
        for i in weather_train.loc[i].air_temperature:
            if type(i) == np.float64:
                temp2 += i
                temp3 += 1
        temp1.append(temp2/temp3)


# In[ ]:


weather_train['air_temperature'] = temp1


# In[ ]:


weather_train['air_temperature'].isna().sum()


# **Missing values in rest of the columns of weather_train**

# For these columns similar steps will be taken, as air_temperature

# In[ ]:


weather_train['dew_temperature'].isna().sum()


# In[ ]:


weather_train['cloud_coverage'].isna().sum()


# In[ ]:


weather_train['sea_level_pressure'].isna().sum()


# In[ ]:


weather_train['wind_direction'].isna().sum()


# In[ ]:


weather_train['wind_speed'].isna().sum()


# In[ ]:


for j in ['dew_temperature','wind_speed','wind_direction','sea_level_pressure','cloud_coverage']:
    temp = weather_train.groupby(['site_id','date','time'])[j].agg(pd.Series.mode)
    temp_list = []
    for i in weather_train.index:
        if np.isnan(weather_train.loc[i][j]):
            try:
                temp1 = temp[[weather_train.loc[i].site_id,weather_train.loc[i].date,weather_train.loc[i].time]]
                temp_list.append(temp1)
            except:
                temp_list.append(np.nan)
        else:
            temp_list.append(weather_train.loc[i][j])
    weather_train[j] = temp_list
    temp1 = []
    for i in weather_train.index:
        if type(weather_train.loc[i].air_temperature) == np.float64:
            temp = weather_train.loc[i].air_temperature
            temp1.append(temp)
        else:
            temp2 = 0
            temp3 = 0
            for i in weather_train.loc[i].air_temperature:
                if type(i) == np.float64:
                    temp2 += i
                    temp3 += 1
            temp1.append(temp2/temp3)
    weather_train[j] = temp1


# In[ ]:


weather_train['precip_depth_1_hr'].isna().sum()


# In[ ]:


del weather_train['precip_depth_1_hr']


# In[ ]:


weather_train.to_csv('cleaned_weather_train.csv',index=False)


# In[ ]:


del weather_train
gc.collect()


# **Part 2 of this notebook, that contains cleanup for weather_test files, it is handled in different kernel as its large size was causing problem**
# **Link to notebook - [Click here](https://www.kaggle.com/rohan9889/handling-missing-data-ashrae-pt2)**
