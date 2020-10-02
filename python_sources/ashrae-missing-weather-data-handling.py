#!/usr/bin/env python
# coding: utf-8

# ## About
# 
# Energy consumption is heavily dependent on weather conditions. This weather dataset has lots of missing weather information. This kernel is dedicated to fix this issue.
# 

# ## Changelog
# 
# * **V5** - Unused variable removed.
# * **V4** - Super Fast now. I was using previous version in my model and it was very slow so I decided to make it fast.
# * **V3** - Bug Fix - Filler method is correctly specified.
# 

# In[ ]:


import pandas as pd
import numpy as np
import warnings
import datetime

warnings.filterwarnings('ignore')
                        
DATA_PATH = '../input/ashrae-energy-prediction/'


# ## Weather Train Data Set

# In[ ]:


weather_df = pd.read_csv(DATA_PATH + 'weather_train.csv')


# ## Utility Functions

# In[ ]:


def missing_statistics(df):    
    statitics = pd.DataFrame(df.isnull().sum()).reset_index()
    statitics.columns=['COLUMN NAME',"MISSING VALUES"]
    statitics['TOTAL ROWS'] = df.shape[0]
    statitics['% MISSING'] = round((statitics['MISSING VALUES']/statitics['TOTAL ROWS'])*100,2)
    return statitics


# ## Missing Hours
# 
# This csv has hourly weather information for 16 sites in 2016. So this should have 140,544 records (16 x 24 x 366, 2016 is a leap year). But this csv has 139,773 records so 771 hours of data is missing. 

# In[ ]:


time_format = "%Y-%m-%d %H:%M:%S"
start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)
end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)
total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

missing_hours = []
for site_id in range(16):
    site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
    new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
    new_rows['site_id'] = site_id
    weather_df = pd.concat([weather_df,new_rows])

weather_df = weather_df.reset_index(drop=True)    


# ### Count Missing Values

# In[ ]:


missing_statistics(weather_df)


# ## Add Day,Week & Month Columns
# 
# This dataset consits of hourly weather information. So we are going to fill missing values based on below new date features.

# In[ ]:


weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
weather_df["day"] = weather_df["datetime"].dt.day
weather_df["week"] = weather_df["datetime"].dt.week
weather_df["month"] = weather_df["datetime"].dt.month


# ## Reset Index for Fast Update

# In[ ]:


weather_df = weather_df.set_index(['site_id','day','month'])


# ### Fill Air Temperature
# 
# Ides is to fill missing air temperature with mean temperature of day of the month. Each month comes in a season and temperature varies lots in a season. So filling with yearly mean value is not a good idea.

# In[ ]:


air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
weather_df.update(air_temperature_filler,overwrite=False)


# ### Fill Cloud Coverage
# 
# Almost 50% data is missing. And data is missing for most of days and even many consecutive days. So, first, calculate mean cloud coverage of day of the month and then fill rest missing values with  last valid observation.

# In[ ]:


# Step 1
cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
# Step 2
cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

weather_df.update(cloud_coverage_filler,overwrite=False)


# ### Fill Dew Temperature

# In[ ]:


due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
weather_df.update(due_temperature_filler,overwrite=False)


# ### Fill Sea level Pressure

# In[ ]:


# Step 1
sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
# Step 2
sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])

weather_df.update(sea_level_filler,overwrite=False)


# ### Fill Wind Direction

# In[ ]:


wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
weather_df.update(wind_direction_filler,overwrite=False)


# ### Fill Wind Speed

# In[ ]:


wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
weather_df.update(wind_speed_filler,overwrite=False)


# ### Fill precip_depth_1_hr
# 
# I noted this value is ignored by most of the kagglers and it has 35% missing data. This value indicates a very light and brief rain or snow, you often won't know it unless you go outside. At a moment, this value will be just a greater than zero but not measureable by rain gauge.

# In[ ]:


# Step 1
precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
# Step 2
precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

weather_df.update(precip_depth_filler,overwrite=False)


# ## Remove Extra Features

# In[ ]:


weather_df = weather_df.reset_index()
weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)


# ### Count Missing Values 

# In[ ]:


missing_statistics(weather_df)


# ## How to Use
# 
# I have put all code in one function so you can just copy and use it.     

# In[ ]:


def fill_weather_dataset(weather_df):
    
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df,new_rows])

        weather_df = weather_df.reset_index(drop=True)           

    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    
    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id','day','month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)

    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler,overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)

    # Step 1
    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler,overwrite=False)

    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)

    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)

    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)
        
    return weather_df


# ## Usages

# In[ ]:


weather_train_df = pd.read_csv(DATA_PATH + 'weather_train.csv')
weather_test_df = pd.read_csv(DATA_PATH + 'weather_test.csv')

weather_train_df = fill_weather_dataset(weather_train_df)
weather_test_df = fill_weather_dataset(weather_test_df)


# ## Discussion
# 
# If you have more ideas to fill these missing values accurately, please share in comments.
