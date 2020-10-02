#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from time import time
import datetime
import gc
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',1500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.model_selection import train_test_split,KFold,GroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from plotly.offline import init_notebook_mode,iplot,plot
import plotly.graph_objects as go
init_notebook_mode(connected=True)
import plotly.figure_factory as ff


# In[ ]:


metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}
metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv",dtype=metadata_dtype)
metadata.info(memory_usage='deep')


# In[ ]:


weather_dtype = {"site_id":"uint8"}
weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv",parse_dates=['timestamp'],dtype=weather_dtype)
weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv",parse_dates=['timestamp'],dtype=weather_dtype)
print (weather_train.info(memory_usage='deep'))
print ("-------------------------------------")
print (weather_test.info(memory_usage='deep'))


# In[ ]:


train_dtype = {'meter':"uint8",'building_id':'uint16','meter_reading':"float32"}
train = pd.read_csv("../input/ashrae-energy-prediction/train.csv",parse_dates=['timestamp'],dtype=train_dtype)
test_dtype = {'meter':"uint8",'building_id':'uint16'}
test_cols_to_read = ['building_id','meter','timestamp']
test = pd.read_csv("../input/ashrae-energy-prediction/test.csv",parse_dates=['timestamp'],usecols=test_cols_to_read,dtype=test_dtype)


# In[ ]:


"""## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df"""


# In[ ]:


"""#reducing memory usage
train= reduce_mem_usage(train)
test= reduce_mem_usage(test)

weather_train = reduce_mem_usage(weather_train)
weather_test = reduce_mem_usage(weather_test)
metadata = reduce_mem_usage(metadata)"""


# In[ ]:


#submission file
Submission = pd.DataFrame(test.index,columns=['row_id'])


# In[ ]:


train.tail()


# In[ ]:


test.head()


# In[ ]:


metadata.head()


# In[ ]:


weather_train.head()


# In[ ]:


weather_test.tail()


# In[ ]:


#checking missing values
missing_weather = pd.DataFrame(weather_train.isna().sum()/len(weather_train),columns=["Weather_Train_Missing_%"])
missing_weather["Weather_Test_Missing_%"] = weather_test.isna().sum()/len(weather_test)
missing_weather


# In[ ]:


#missing metadata
metadata.isna().sum()/len(metadata)


# In[ ]:


#dropped floor count since we have squarefoot
metadata['floor_count_isNa'] = metadata['floor_count'].isna().astype('uint8')
metadata['year_built_isNa'] = metadata['year_built'].isna().astype('uint8')
# Dropping floor_count variable as it has 75% missing values
metadata.drop('floor_count',axis=1,inplace=True)


# In[ ]:


missing_train_test = pd.DataFrame(train.isna().sum()/len(train),columns=["Missing_Pct_Train"])
missing_train_test["Missing_Pct_Test"] = test.isna().sum()/len(test)
missing_train_test


# <h2>Exploring the data<h2>

# In[ ]:


train.tail()
#we will need to combine weather file with train


# In[ ]:


train.describe(include='all')
# Data contains records from 1st Jan to 31st Dec of 2016.
# Data has information about 1448 buildings.
# Data has 4 meter types.
# Some extremely high values in meter reading which can be explored further.


# In[ ]:


train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)
test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)


# In[ ]:


trace1 = go.Bar(x=train['meter'].unique(),y=train['meter'].value_counts().values,marker=dict(color="red"),text='train')
trace2 = go.Bar(x=test['meter'].unique(),y=test['meter'].value_counts().values,marker=dict(color="blue"),text='test')
data=[trace1,trace2]
layout = go.Layout(title='Countplot of meter',xaxis=dict(title='Meter'),yaxis=dict(title='Count'),hovermode='closest')
figure = go.Figure(data=data,layout=layout)
iplot(figure)


# In[ ]:


train[train['building_id'] == 1022]['meter'].unique()
# Like it is mentioned in the competition description, each building may or may not have all 4 meter Id codes.


# In[ ]:


train.groupby('meter')['meter_reading'].agg(['min','max','mean','median','count','std'])
#since min is 0 we must drop these values before training the model


# In[ ]:


for df in [train, test]:
    df['Month'] = df['timestamp'].dt.month.astype("uint8")
    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")
    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")
    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")


# In[ ]:


train[['timestamp','meter_reading']].set_index('timestamp').resample("H")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Hour')
train[['timestamp','meter_reading']].set_index('timestamp').resample("D")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Day')
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Average Meter Reading")
plt.title("Graph of Average Meter Reading")
# meter readings by hour and days..//resample used to sample data


# In[ ]:


meter_Electricity = train[train['meter'] == "Electricity"]
meter_Electricity[['timestamp','meter_reading']].set_index('timestamp').resample("H")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Hour')
meter_Electricity[['timestamp','meter_reading']].set_index('timestamp').resample("D")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Day')
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Average Meter Reading")
plt.title("Graph of Average Meter Readingfor Electricity Meter")
# The increase and decreasing trend can be attributed to the usage during the weekdays and during the weekends when it drops. 


# In[ ]:


meter_ChilledWater = train[train['meter'] == "ChilledWater"]
meter_ChilledWater[['timestamp','meter_reading']].set_index('timestamp').resample("H")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Hour')
meter_ChilledWater[['timestamp','meter_reading']].set_index('timestamp').resample("D")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Day')
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Average Meter Reading")
plt.title("Graph of Average Meter Readingfor ChilledWater Meter")
# Consumption gradually increases and reaches its peak during septembet to November months. 


# In[ ]:


meter_Steam = train[train['meter'] == "Steam"]
meter_Steam[['timestamp','meter_reading']].set_index('timestamp').resample("H")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Hour')
meter_Steam[['timestamp','meter_reading']].set_index('timestamp').resample("D")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Day')
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Average Meter Reading")
plt.title("Graph of Average Meter Readingfor Steam Meter")
# This is almost similar to that of the overall trend. 


# In[ ]:


meter_HotWater = train[train['meter'] == "HotWater"]
meter_HotWater[['timestamp','meter_reading']].set_index('timestamp').resample("H")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Hour')
meter_HotWater[['timestamp','meter_reading']].set_index('timestamp').resample("D")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Day')
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Average Meter Reading")
plt.title("Graph of Average Meter Readingfor HotWater Meter")
# Hot water meter reading is high during the winter months and reduces during the summer months. 


# In[ ]:


train.groupby(['meter','Month'])['meter_reading'].agg(['max','mean','median','count','std'])
# We can see that only Steam meter has very high meter_reading values as compared to other types of meters.
# We can see that the average electricity meter_reading does not vary much across the months.
# Average Hot Water meter_reading is relatively less from April to October Months.
# Average Steam meter_reading is way higher from March to June as compared to the other months.


# In[ ]:


train.groupby(['meter','DayOfWeek'])['meter_reading'].agg(['max','mean','median','count','std'])
# Average meter_reading of Steam type of meter is higher as compared to the other meter types.


# In[ ]:


summary=train.groupby('building_id')['meter_reading'].agg(['count','min','max','mean','median','std'])


# In[ ]:


#building wise summary
summary.sort_values(['mean'],ascending=False).head(20)


# In[ ]:


train[train['building_id'] == 1099]['meter_reading'].describe()


# In[ ]:


plt.hist(train[train['building_id'] == 1099]['meter_reading'])


# In[ ]:


train['meter_reading'].describe()


# In[ ]:


sns.distplot(np.log1p(train['meter_reading']),kde=False)
plt.title("Distribution of Log of Meter Reading Variable")
# Lot of 0 values as can be seen from the distribution


# In[ ]:


sns.boxplot(np.log1p(train[train['meter'] == "Electricity"]['meter_reading']))
plt.title("Boxplot of Meter Reading Variable for the Meter Type: Electricity")
# We can see a few outliers here.


# In[ ]:


sns.boxplot(np.log1p(train[train['meter'] == "ChilledWater"]['meter_reading']))
plt.title("Boxplot of Meter Reading Variable for the Meter Type: Electricity")
# Not many outliers here. 


# In[ ]:


sns.boxplot(np.log1p(train[train['meter'] == "HotWater"]['meter_reading']))
plt.title("Boxplot of Meter Reading Variable for the Meter Type: Electricity")


# In[ ]:


sns.boxplot(np.log1p(train[train['meter'] == "Steam"]['meter_reading']))
plt.title("Boxplot of Meter Reading Variable for the Meter Type: Electricity")


# In[ ]:


sns.distplot(np.log1p(train[train['meter'] == "Electricity"]['meter_reading']),kde=False)
plt.title("Distribution of Meter Reading per MeterID code: Electricity")
#same is case for other meter types we will neen to remove zero values


# In[ ]:


metadata.info()
# Missing values in year_built and floor_count variables.


# In[ ]:


metadata.head()


# In[ ]:


cols = ['site_id','primary_use','building_id','year_built']
for col in cols:
    print ("Number of Unique Values in the {} column are:".format(col),metadata[col].nunique())


# In[ ]:


cols = ['site_id','primary_use','year_built']
for col in cols:
    print ("Unique Values in the {} column are:".format(col),metadata[col].unique())
    print ("\n")


# In[ ]:


trace1 = go.Bar(x=metadata['site_id'].unique(),y=metadata['site_id'].value_counts().values,marker=dict(color="blue"))
data=[trace1]
layout = go.Layout(title='Countplot of site_id variable',xaxis=dict(title='site_id'),yaxis=dict(title='Count'),hovermode='closest')
figure = go.Figure(data=data,layout=layout)
iplot(figure)


# In[ ]:


trace1 = go.Bar(x=metadata['primary_use'].unique(),y=metadata['primary_use'].value_counts().values,marker=dict(color="rgb(55, 83, 109)"))
data=[trace1]
layout = go.Layout(title='Countplot of primary_use variable',xaxis=dict(title='primary_use'),yaxis=dict(title='Count'),hovermode='closest')
figure = go.Figure(data=data,layout=layout)
iplot(figure)
# Education, Office, Entertainment/Public Assembly, Public Services, Lodging/Residential form the bulk of Primary Use


# In[ ]:


#to combine categories or not???
#metadata['primary_use'].replace({"Healthcare":"Other","Parking":"Other","Warehouse/storage":"Other","Manufacturing/industrial":"Other",
 #                               "Retail":"Other","Services":"Other","Technology/science":"Other","Food sales and service":"Other",
  #                              "Utility":"Other","Religious worship":"Other"},inplace=True)


# In[ ]:


metadata['square_feet'].describe()


# In[ ]:


metadata['square_feet'] = np.log1p(metadata['square_feet'])


# In[ ]:


sns.distplot(metadata['square_feet'])
plt.title("Distribution of Square Feet variable of Metadata Table")
plt.xlabel("Area in Square Feet")
plt.ylabel("Frequency")
# Looks like a normal distribution distribution


# In[ ]:


sns.boxplot(metadata['square_feet'])
plt.title("Box Plot of Square Feet Variable")
# There are a few outliers visible


# In[ ]:


metadata.groupby('primary_use')['square_feet'].agg(['mean','median','count']).sort_values(by='count')
# Parking has the highest average are although the count is less.
# Education has the highest count as can be seen in the countplot above.


# In[ ]:


trace1 = go.Bar(x=metadata['year_built'].unique(),y=metadata['year_built'].value_counts().values,marker=dict(color="blue"))
data=[trace1]
layout = go.Layout(title='Countplot of year_built variable',xaxis=dict(title='year_built'),yaxis=dict(title='Count'),hovermode='closest')
figure = go.Figure(data=data,layout=layout)
iplot(figure)


# In[ ]:


metadata.describe()


# In[ ]:


#should blank year be filled with 1976??
metadata['year_built'].fillna(1976, inplace=True)
metadata['year_built'] = metadata['year_built'].astype('int16')


# In[ ]:





# <h2>Weather Data<h2>

# In[ ]:


weather_train.head()


# In[ ]:


weather_train.isna().sum()/len(weather_train)


# In[ ]:


weather_train.info(memory_usage='deep')


# In[ ]:


weather_train[['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_speed']].describe()
#Missing values in air_temperature, cloud_coverage, dew_temperature, precip_depth_1_hr, sea_level_pressure, wind_speed variables
#There are negative values in air_temperature, dew_temperature and precip_depth_1_hr variables.
#Looks like there are outliers in precip_depth_1_hr variable (can be guessed from Max value).
#min value of wind_speed as 0 does not make any sense.
#Only temperature would be crucial


# In[ ]:


weather_train['timestamp'].describe()
# This data is from 1st Jan to 31st Dec 2016, similar to the timestamp of the training data


# In[ ]:


cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_speed']
for ind,col in enumerate(weather_train[cols]):
    plt.figure(ind)
    sns.distplot(weather_train[col].dropna())


# Distribution of sea_level_pressure looks like a normal distribution.
# 
# Lot of 0 values in precip_depth_1_hr variable.
# 
# Wind_Speed distribution looks like positively skewed.
# 
# Dew Temperature looks like a Negatively skewed distribution.
# 
# Cloud_Coverage takes distinct values unlike these other variables.

# In[ ]:


cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_speed']
for ind,col in enumerate(weather_train[cols]):
    plt.figure(ind)
    sns.boxplot(weather_train[col].dropna())


# In[ ]:


weather_test.info(memory_usage='deep')


# In[ ]:


weather_test.info(memory_usage='deep')


# In[ ]:


def fill_weather_dataset(weather_df):
    
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"

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


# In[ ]:


#filling blank values of weather
weather_train = fill_weather_dataset(weather_train)
weather_test = fill_weather_dataset(weather_test)


# In[ ]:


cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']
for col in cols:
    print (" Minimum Value of {} column is {}".format(col,weather_train[col].min()))
    print (" Maximum Value of {} column is {}".format(col,weather_train[col].max()))
    print ("----------------------------------------------------------------------")


# In[ ]:


for df in [weather_train,weather_test]:
    df['air_temperature'] = df['air_temperature'].astype('float32')
    df['cloud_coverage'] = df['cloud_coverage'].astype('float16')
    df['dew_temperature'] = df['dew_temperature'].astype('float16')
    df['precip_depth_1_hr'] = df['precip_depth_1_hr'].astype('float32')
    df['sea_level_pressure'] = df['sea_level_pressure'].astype('float32')
    df['wind_direction'] = df['wind_direction'].astype('float32')
    df['wind_speed'] = df['wind_speed'].astype('float16')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = pd.merge(train,metadata,on=\'building_id\',how=\'left\')\ntest  = pd.merge(test,metadata,on=\'building_id\',how=\'left\')\nprint ("Training Data Shape {}".format(train.shape))\nprint ("Testing Data Shape {}".format(test.shape))\ngc.collect()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = pd.merge(train,weather_train,on=[\'site_id\',\'timestamp\'],how=\'left\')\ntest  = pd.merge(test,weather_test,on=[\'site_id\',\'timestamp\'],how=\'left\')\nprint ("Training Data Shape {}".format(train.shape))\nprint ("Testing Data Shape {}".format(test.shape))\ngc.collect()')


# In[ ]:


#since we have not filed blank years this might give errors/wrong values
for df in [train,test]:
    df['square_feet'] = df['square_feet'].astype('float16')
    df['Age'] = df['timestamp'].dt.year - df['year_built']


# In[ ]:


# As per the discussion in the following thread, https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117083, there is some discrepancy in the meter_readings for different ste_id's and buildings. It makes sense to delete them
idx_to_drop = list((train[(train['site_id'] == 0) & (train['timestamp'] < "2016-05-21 00:00:00")]).index)
print (len(idx_to_drop))
train.drop(idx_to_drop,axis='rows',inplace=True)


# In[ ]:


# dropping all the electricity meter readings that are 0, after considering them as anomalies.
idx_to_drop = list(train[(train['meter'] == "Electricity") & (train['meter_reading'] == 0)].index)
print(len(idx_to_drop))
train.drop(idx_to_drop,axis='rows',inplace=True)


# In[ ]:


idx_to_drop = list((train[(train['building_id']==1099)&(train['meter_reading'] > 30000)&(train['meter'] == "Steam")]).index)
print (len(idx_to_drop))
train.drop(idx_to_drop,axis='rows',inplace=True)


# In[ ]:


# Converting the dependent variable to logarithmic scale
train['meter_reading'] = np.log1p(train['meter_reading'])


# In[ ]:


get_ipython().run_cell_magic('time', '', "mean_meter_reading_per_building = train.groupby('building_id')['meter_reading'].mean()\ntrain['mean_meter_reading_per_building'] = train['building_id'].map(mean_meter_reading_per_building)\nmedian_meter_reading_per_building = train.groupby('building_id')['meter_reading'].median()\ntrain['median_meter_reading_per_building'] = train['building_id'].map(median_meter_reading_per_building)\nstd_meter_reading_per_building = train.groupby('building_id')['meter_reading'].std()\ntrain['std_meter_reading_per_building'] = train['building_id'].map(std_meter_reading_per_building)\n\nmean_meter_reading_per_dayofweek = train.groupby('DayOfWeek')['meter_reading'].mean()\ntrain['mean_meter_reading_per_dayofweek'] = train['DayOfWeek'].map(mean_meter_reading_per_dayofweek)\nmedian_meter_reading_per_dayofweek = train.groupby('DayOfWeek')['meter_reading'].median()\ntrain['median_meter_reading_per_dayofweek'] = train['DayOfWeek'].map(median_meter_reading_per_dayofweek)\nstd_meter_reading_per_dayofweek = train.groupby('DayOfWeek')['meter_reading'].std()\ntrain['std_meter_reading_per_dayofweek'] = train['DayOfWeek'].map(std_meter_reading_per_dayofweek)\n\n\nmean_meter_reading_per_meter = train.groupby('meter')['meter_reading'].mean()\ntrain['mean_meter_reading_per_meter'] = train['meter'].map(mean_meter_reading_per_meter)\nmedian_meter_reading_per_meter = train.groupby('meter')['meter_reading'].median()\ntrain['median_meter_reading_per_meter'] = train['meter'].map(median_meter_reading_per_meter)\nstd_meter_reading_per_meter = train.groupby('meter')['meter_reading'].std()\ntrain['std_meter_reading_per_meter'] = train['meter'].map(std_meter_reading_per_meter)\n\n\nmean_meter_reading_per_month = train.groupby('Month')['meter_reading'].mean()\ntrain['mean_meter_reading_per_month'] = train['Month'].map(mean_meter_reading_per_month)\nmedian_meter_reading_per_month = train.groupby('Month')['meter_reading'].median()\ntrain['median_meter_reading_per_month'] = train['Month'].map(median_meter_reading_per_month)\nstd_meter_reading_per_month = train.groupby('Month')['meter_reading'].std()\ntrain['std_meter_reading_per_month'] = train['Month'].map(std_meter_reading_per_month)\n\n\ntest['mean_meter_reading_per_building'] = test['building_id'].map(mean_meter_reading_per_building)\ntest['median_meter_reading_per_building'] = test['building_id'].map(median_meter_reading_per_building)\ntest['std_meter_reading_per_building'] = test['building_id'].map(std_meter_reading_per_building)\n\ntest['mean_meter_reading_per_dayofweek'] = test['year_built'].map(mean_meter_reading_per_dayofweek)\ntest['median_meter_reading_per_dayofweek'] = test['year_built'].map(median_meter_reading_per_dayofweek)\ntest['std_meter_reading_per_dayofweek'] = test['year_built'].map(std_meter_reading_per_dayofweek)\n\ntest['mean_meter_reading_per_meter'] = test['meter'].map(mean_meter_reading_per_meter)\ntest['median_meter_reading_per_meter'] = test['meter'].map(median_meter_reading_per_meter)\ntest['std_meter_reading_per_meter'] = test['meter'].map(std_meter_reading_per_meter)\n\ntest['mean_meter_reading_per_month'] = test['primary_use'].map(mean_meter_reading_per_month)\ntest['median_meter_reading_per_month'] = test['primary_use'].map(median_meter_reading_per_month)\ntest['std_meter_reading_per_month'] = test['primary_use'].map(std_meter_reading_per_month)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for df in [train, test]:\n    df[\'mean_meter_reading_per_building\'] = df[\'mean_meter_reading_per_building\'].astype("float16")\n    df[\'median_meter_reading_per_building\'] = df[\'mean_meter_reading_per_building\'].astype("float16")\n    df[\'std_meter_reading_per_building\'] = df[\'std_meter_reading_per_building\'].astype("float16")\n    \n    df[\'mean_meter_reading_per_meter\'] = df[\'mean_meter_reading_per_meter\'].astype("float16")\n    df[\'median_meter_reading_per_meter\'] = df[\'median_meter_reading_per_meter\'].astype("float16")\n    df[\'std_meter_reading_per_meter\'] = df[\'std_meter_reading_per_meter\'].astype("float16")\n    \n    df[\'mean_meter_reading_per_dayofweek\'] = df[\'mean_meter_reading_per_dayofweek\'].astype("float16")\n    df[\'median_meter_reading_per_dayofweek\'] = df[\'median_meter_reading_per_dayofweek\'].astype("float16")\n    df[\'std_meter_reading_per_dayofweek\'] = df[\'std_meter_reading_per_dayofweek\'].astype("float16")\n    \n    df[\'mean_meter_reading_per_month\'] = df[\'mean_meter_reading_per_month\'].astype("float16")\n    df[\'median_meter_reading_per_month\'] = df[\'median_meter_reading_per_month\'].astype("float16")\n    df[\'std_meter_reading_per_month\'] = df[\'std_meter_reading_per_month\'].astype("float16")\n    \n    df[\'Age\'] = df[\'Age\'].astype(\'uint8\')\ngc.collect()')


# In[ ]:


train.drop(['timestamp','year_built'],axis=1,inplace=True)
test.drop(['timestamp','year_built'],axis=1,inplace=True)


# In[ ]:


print (train.shape, test.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'le = LabelEncoder()\n\ntrain[\'meter\']= le.fit_transform(train[\'meter\']).astype("uint8")\ntest[\'meter\']= le.fit_transform(test[\'meter\']).astype("uint8")\ntrain[\'primary_use\']= le.fit_transform(train[\'primary_use\']).astype("uint8")\ntest[\'primary_use\']= le.fit_transform(test[\'primary_use\']).astype("uint8")')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Let's check the correlation between the variables and eliminate the one's that have high correlation\n# Threshold for removing correlated variables\nthreshold = 0.9\n\n# Absolute value correlation matrix\ncorr_matrix = train.corr().abs()\ncorr_matrix.head()")


# In[ ]:


# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()


# In[ ]:


# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))
print ("Following columns can be dropped {}".format(to_drop))


# In[ ]:


train.drop(to_drop,axis=1,inplace=True)
test.drop(to_drop,axis=1,inplace=True)


# In[ ]:


y = train['meter_reading']
train.drop('meter_reading',axis=1,inplace=True)


# In[ ]:


categorical_cols = ['building_id','Month','meter','Hour','primary_use','DayOfWeek','DayOfMonth','floor_count_isNa']


# In[ ]:


params = {'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'objective': 'regression',
          'max_depth': 11,
          'learning_rate': 0.15,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'rmse',
          "verbosity": -1,
          'reg_alpha': 0.5,
          'reg_lambda': 0.5,
          'random_state': 47,
          "num_leaves": 31}

kf = KFold(n_splits=3)
models = []
for train_index,test_index in kf.split(train):
    train_features = train.iloc[train_index]
    train_target = y.iloc[train_index]
    
    test_features = train.iloc[test_index]
    test_target = y.iloc[test_index]
    
    d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_cols, free_raw_data=False)
    d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_cols, free_raw_data=False)
    
    model = lgb.train(params, train_set=d_training, num_boost_round=2000, valid_sets=[d_training,d_test], verbose_eval=100, early_stopping_rounds=50)
    models.append(model)
    gc.collect()


# In[ ]:


ser1 = pd.DataFrame(models[0].feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')
ser1['Importance'].plot(kind='bar',figsize=(10,6))


# In[ ]:


ser2 = pd.DataFrame(models[1].feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')
ser2['Importance'].plot(kind='bar',figsize=(10,6))


# In[ ]:


ser3 = pd.DataFrame(models[2].feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')
ser3['Importance'].plot(kind='bar',figsize=(10,6))


# In[ ]:




