#!/usr/bin/env python
# coding: utf-8

# ![Save Energy](https://isoilonline.com/wp-content/uploads/energy-efficiency.jpg)

# # Quick Intro
# A quick intro to the Kernel. The competition seeks to retroactively measure future hourly energy consumption for 2 years based on past one year consumption. The training data for hourly energy consumption is collected from thousands of buildings across 16 sites measuring electricity, chilled water, steam and hotwater consumption. To aid in the analysis building attributes such as square feet, floor count etc. are collected as well as various ambient weather conditions at the 16 sites in which the buildings under study are located. 
# 
# In this Kernel, we will do EDA on various aspects of the data such as energy consumption by building, site, pirmary use of building, type of energy used as well as the pattern of weather parameters such temperature, pressure, wind speed, cloud cover etc. and try to record what is the importance of each of these features so that we can explore options for feature selection and feature engineering as the road to prediction modeling.

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


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Import and Preview Data

# In[ ]:


building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')
train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')
test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')


# ## Preview Building_Metadata table

# In[ ]:


print('The shape of Building Metadata is :',building_metadata.shape)
building_metadata.head()


# ## Preview Weather_train data

# In[ ]:


print('The shape of weather_train data is :',weather_train.shape)
weather_train.head()


# ## Preview train data

# In[ ]:


print('The shape of train data is :',train.shape)
train.head()


# ## Preview weather_test data

# In[ ]:


print('The shape of weather_test data is :',weather_test.shape)
weather_test.head()


# ## Preview test data

# In[ ]:


print('The shape of test data is :',test.shape)
test.head()


# # Key observations
# ## Key observations on building metadata 
# ### How many buildings are under the study?

# In[ ]:


print('Number of Buildings :', len(building_metadata.building_id.unique()))


# ### How many sites are under study and how many buildings under each site?

# In[ ]:


print('Number of Sites and number of Buildings in Each site')
print(building_metadata.site_id.unique())
print(building_metadata.site_id.value_counts().sort_index())


# So we have 1449 buildings in all spread across 16 sites.

# ### What are the buildings used for?

# In[ ]:


building_count = building_metadata.primary_use.value_counts()
print('Number of Buildings by Primary Use:')
plt.figure(figsize=(15,3))
sns.barplot(building_count.index,building_metadata.primary_use.value_counts())
plt.xticks(rotation = 90)
plt.ylabel('Number of Buildings')


# ## Reduce Memory Consumption
# #### source: https://www.kaggle.com/kernels/scriptcontent/3684066/download
# The train and test sets have millions of rows with a huge size in GBs. We will reduce the size of the dataframes for better performance.

# In[ ]:


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
    return df


# In[ ]:


## Reducing memory of the data frames
building_metadata = reduce_mem_usage(building_metadata)
train = reduce_mem_usage(train)
weather_train = reduce_mem_usage(weather_train)
weather_test = reduce_mem_usage(weather_test)
test = reduce_mem_usage(test)


# # Merge Datasets and extract Date time features

# In[ ]:


# First merge train and building data
train = pd.merge(train,building_metadata,how = 'left')           
print(train.shape)
train.head()

# Now Merge train_building with weather_train data
train = pd.merge(train,weather_train, on = ['site_id','timestamp'], how = 'left')
print(train.shape)
train.head()


# In[ ]:


# First merge test and building data
test = pd.merge(test,building_metadata,how = 'left')           
print(test.shape)
test.head()

# Now Merge test_building with weather_test data
test = pd.merge(test,weather_test, on = ['site_id','timestamp'], how = 'left')
print(test.shape)
test.head()


# Convert 'timestamp' to datetime object

# In[ ]:


train['timestamp'] = pd.to_datetime(train.timestamp)
test['timestamp'] = pd.to_datetime(test.timestamp)


# ## Missing value Report for final Merged Train Data

# In[ ]:


def report_missing_data(df):
    print('Total Number of rows :', len(df))
    for column in df.columns:
        print(column,':', 'Missing rows:', sum(df[column].isnull()), '|', '% Missing: {:.2f}'.format(sum(df[column].isnull())*100/len(df)),'%')
report_missing_data(train)


# ## Extract Date time features for Train set

# In[ ]:


# Get month, day, weekday,weekday name, hour etc. from the datetime object - timestamp.
train['month'] = train.timestamp.dt.month
train['day'] = train.timestamp.dt.day
train['weekday'] = train.timestamp.dt.weekday
train['hour'] = train.timestamp.dt.hour
train['weekday_name'] = train.timestamp.dt.weekday_name

# Get meter names from codes for better understanding
meter_dict = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
train['meter_name'] = train.meter.map(meter_dict)

# Calculate elapsed years for each building
train['meter_year'] = train.timestamp.dt.year 
train['elapsed_years'] = train.meter_year - train.year_built

train.head()


# In[ ]:


# Reduce the size of merged train data
train = reduce_mem_usage(train)


# ## Check the distribution of the target variable

# In[ ]:


sns.distplot(train.meter_reading, kde = False)


# There are too many meter readings with value of zero and hence confounding actual distribution. We can deal with such outliers through two ways
# 
# * Outlier removal, based on outlier thresholds or setting IQR thresholds above or below which observations are treated as outliers.
# * Apply Log Transformation to bring the points closer so that the outliers are squished in, and other observations become visible.
# 
# We will use the Log transform approach and re-display the target variable - 'meter_reading' distribution. 

# In[ ]:


# plot hist of log transformed target variable
train['log_meter_reading'] = np.log(train.meter_reading + 1)
sns.distplot(train.log_meter_reading)


# Now we have managed to bring the target variable into a normal distribution. This will be useful when running Linear Regression models for predicting meter readings.
# Now Lets try to understand what are the factors influencing Energy consumption. We will analyze in the following angles:
# 
# 

# # Analyze Energy Consumption
# We will focus on analyzing energy consumption across different dimensions - 
# 
# 1. **By Primary Use** - Which types of buildings are more frequent in this study? which of them consume more energy on an average
# 2. **By Meter Type** - Which meters are more installed in the buildings under this study? Which of them show more average energy consumption?
# 3. **By Site Id** - Which Sites consume more energy? Some sites have more buildings and hence can contribute more to energy consumption. Some sites may use more of a particular energy meter which has its own consumption characteristics.

# ## 1.1 Analyze Energy Consumption based on Primary Use

# In[ ]:


plt.subplot(1,3,2)
train.groupby('primary_use')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings by Primary use')
plt.subplot(1,3,1)
train.groupby('primary_use')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption by Primary use')
plt.subplot(1,3,3)
train.groupby('primary_use')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')
plt.tight_layout()


# The ideal candidates for piloting energy consumption initiatives could be those which have more energy consumption across lesser number of buildings.
# 
# **Education and Services have the highest mean energy consumption.** While Education also has the maximum total as well as mean energy consumption, Services while having few buildings have second highest mean consumption. **Service buildings while being fewer in number can contribute to more energy savings**.All other buildings have far less mean consumption.
# 
# Lets look deeper into the **'mean consumption by primary use'** plot and try to break it down based on **which meter types contribute to more consumption** in each of these primary use building type.

# ## 1.2 Energy Consumption by Primary Use by Meter Type

# In[ ]:


#train.groupby(['primary_use','meter'])['meter_reading'].mean().sort_values(ascending = False).reset_index().plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')
pivot_df = train.groupby(['primary_use','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()
pivot_df.head()
pivot_df = pivot_df.pivot(index='primary_use', columns='meter_name', values='meter_reading')
pivot_df['AllMeters'] = pivot_df.sum(axis = 1)
pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Hourly Energy consumption by Primary use by meter', stacked = True)
plt.tight_layout()


# Steam is more used than other meter types in Education, Services, Public Services, Health care, office, parking, Food sales, lodging, manufacturing etc. Also Looks like Steam consumption is more than 90% for the highest energy consuming buildings like Education, Services etc. 
# 
# Though Electricity meters are installed in maximum number of buildings, they seem to show much lower hourly average consumption. hot water seems to be the least used meter.
# Chilled water is more used in 
# 
# Lets look at the overall trend for the mean consumption by meter type - we would expect steam to be the most dominating meter type.

# ## 2.1 Analyze Energy Consumption based on Meter used

# In[ ]:


#plt.figure(figsize=(30,10))
plt.subplot(1,3,2)
train.groupby('meter_name')['building_id'].count().sort_values(ascending = False).plot(kind = 'bar', figsize = (15,6), color = 'purple', title = 'Count of Meter Type')
plt.subplot(1,3,1)
train.groupby('meter_name')['meter_reading'].sum().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Total Energy consumption by Meter Type' )
plt.subplot(1,3,3)
train.groupby('meter_name')['meter_reading'].mean().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Mean Energy consumption by Meter Type' )
plt.tight_layout()


# Electricity meters are** 3 times more used than chilled water **and **4 times more used than steam** and **6 times more used than hotwater**.
# 
# Inspite of that, **Steam and Chilled water seem to be showing highest mean energy consumption** in terms of Total as well as mean hourly energy consumption. **Electricity meter has the lowest mean consumption** while having total consumption slightly higher than hot water due to more meters being installed.
# 
# Given the huge margin between Steam and Other Meter types, **the overall trend for energy consumption will mostly likely be driven by Steam energy meters**.

# ## 3.1 Analyze mean energy consumption by time - by hours, days and months

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# ## 3.2 Analyze mean energy consumption by time - by hours, days and months by **'meter type'**
# By breaking down the overall mean energy consumption trend by 'meter type' we can get insight on how different forms of energy are being consumed through the year. We saw earlier that Steam meter type has the highest total and mean energy consumption, so we expect the trend for overall energy usage and that for steam meters should be pretty identical.

# In[ ]:


meter_dict = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
# Plot energy consumption by meter
plt.figure(figsize = (20,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Hour')
    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Day')
    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Month')
    #plt.title('No 1099: Energy consumption by time by meter type')
    plt.legend()
    plt.tight_layout()
plt.show()


# **Analysis of Meter Types:**
# 
# 1. **Steam** : Closely mirrors the overall trend infact driving the overall trend - June Mid to Nov Mid - a big drop. Then a spike in Nov Mid. The scale of consumption in 10000's.
# 2. **Electricity**: Constant trend from Jan to May after which the consumption peaks to a new constant average - 15% higher than before. The scale of consumption is in smaller 100's
# 3. **Chilled water**: Gradual increase in consumption from Jan to Sep and then a spike in Sep/Oct before finally dropping to march levels by Dec. Probably this is end of summer and moving into winter.The scale of consumption is in 1000's
# 4. **Hot water**: Peak at start of year and continuously falling to a trough (20% of peak) by June-July and then again rising to original peak by end of year. Probably this is summer time when hot water is not needed that much. Scale is in larger 100's to lower 1000's.
# 
# #### Given that the trend is completely different for each meter type, should we be predicting consumption for each meter type separately?

# So once building 1099 was removed, the steam energy trend follows a more continuous pattern starting high at Jan, reducing gradually to a low between June to Sep and climbing up again back to jan levels by the end of the year. This is also the same trend for hotwater usage - kinda nice given steam and hotwater are close cousins:)

# ## 3.3 Analyze Hourly Mean Energy Consumption by Site Id

# In[ ]:


plt.subplot(1,3,1)
train.groupby('site_id')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption in sites 0 to 15')
plt.subplot(1,3,2)
train.groupby('site_id')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings in sites 0 to 15')
plt.subplot(1,3,3)
train.groupby('site_id')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption in sites 0 to 15')


# 6 Sites - 13, 9, 2, 14, 3 and 15 are having about twice the number of buildings as others. Site 13 in particular seems to be 10x as compared to the next highest energy consuming site. We would expect the overall mean energy consumption pattern to be driven by site 13.

# ## 3.4 Check Time Series Hourly and Daily energy consumption by Site Id

# In[ ]:


# Plot energy consumption by site
plt.figure(figsize = (20,20))
for i in range(16):
    plt.subplot(8,2,i+1)
    train[train.site_id == i][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = 'Site {}'.format(i), label = 'By Hour')
    train[train.site_id == i][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(title = 'Site {}'.format(i), label = 'By Day')
    plt.legend()
    plt.tight_layout() # Add tight layout in loop to prevent overlapping text
plt.show()


# Bingo! The trend for Site 13 and for overall trend seems to be identical.I bet Site 13 should have more of Steam consumption as the patterns for overall trend, Steam and Site 13 all look alike. 
# 
# 

# Lets look at a few sites which are interesting
# 
# 1. Site 6 - Major consumption happens only in Sep and Oct. looking at the scale of consumption, it could be only because of steam. 
# 2. Site 8 - pretty normal trend - scale indicates electricity meter usage
# 3. Site 9 - mostly constant low usage (could be electricity) with some peaks in between (could be chilled water or steam)

# In[ ]:


sites = [6,8,9]
for site in sites:
    print('site {}'.format(site), train[train.site_id == site].groupby('meter_name')['meter_reading'].sum())


# ## 3.5 Energy consumption by Site by meter type

# In[ ]:


pivot_df = train.groupby(['site_id','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()
pivot_df.head()
pivot_df = pivot_df.pivot(index='site_id', columns='meter_name', values='meter_reading')
pivot_df['AllMeters'] = pivot_df.sum(axis = 1)
pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)
plt.tight_layout()


## Show values in bar plot
#ax = pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)
#for p in ax.patches:
 #   ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#plt.tight_layout()    


# We can clearly see that Except for Site 13 all other sites do not have mean steam energy usage that much. That is why other sites show a different trend from the Steam meter consumption trend. 

# Finally lets get down to building level. Lets check the general trend of energy consumption by buildings.

# In[ ]:


train.groupby('building_id')['meter_reading'].mean().plot(figsize = (20,6), color = 'green', title = 'Mean Energy consumption in all 1449 buildings')


# Looks like there is one outlier which is dwarfing all other buildings energy consumption. lets sort by ascending order of mean energy consumption by building.

# In[ ]:


train.groupby('building_id')['meter_reading'].mean().sort_values(ascending = False)[:10]


# AS we can see building 1099 has consumption in **million units** while others are in **10,000 units**. Lets see which meters are used in building 1099.

# In[ ]:


train[train.building_id == 1099].groupby('meter_name')['meter_reading'].mean().plot(kind = 'bar', figsize = (6,4), color = 'green', title = 'Mean Energy consumption for Building 1099 by meter')
print('% Steam Consumption for Building 1099 out of total energy consumption is :',(train[train.building_id == 1099].groupby('meter_name')['meter_reading'].mean()[1] / train[train.building_id == 1099].groupby('meter_name')['meter_reading'].mean().sum())*100,'%')


# Literally all of the energy consumed by building 1099 is steam.

# ### What is the mean % consumption of Steam for Building 1099 vs Steam consumption for all buildings? 
# 

# In[ ]:


steam_1099 = train[(train.building_id == 1099) & (train.meter == 2)]['meter_reading'].sum()
steam_others = train[(train.building_id != 1099) & (train.meter == 2)]['meter_reading'].sum()
steam_total = train[train.meter == 2]['meter_reading'].sum()
print('% of Steam Consumption for Building 1099 out of total for all buildings',100*steam_1099/steam_total)


# Quite clearly Building 1099 is dominating the overall steam consumption.Hence it may be better to remove just Building 1099 rather than remove entire steam consumption to understand overall energy usage trend.

# ### How does Site 13 energy consumption look after removing building 1099
# 

# In[ ]:


plt.figure(figsize = (20,10))
train[train.building_id != 1099][train.site_id == 13][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = 'Site 13 - No 1099 - Energy trend', label = 'By Hour')
train[train.building_id != 1099][train.site_id == 13][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(label = 'By Day')
plt.legend()


# This is a more continuous trend for Site 13 without the outlier building 1099.

# Lets filter building 1099 out to get a proper trend of mean energy consumption for all other buildings.

# In[ ]:


train[train.building_id != 1099].groupby('building_id')['meter_reading'].mean().plot(figsize = (20,6), color = 'green', title = 'Mean Energy consumption in all buildings except Building 1099')


# Now we are able to capture the trend of mean hourly energy consumption across buildings in a better way.
# 
# Given Steam meter distorts the whole picture of energy consumption so much, we can exclude outliers and relook at the energy trends.

# ### Energy consumption trend just for Steam meters

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
train[train.meter == 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train[train.meter == 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train[train.meter == 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('Only Steam consumption : Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# ### Energy consumption trend just for Steam meters - outlier Building 1099 removed

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
train[(train.meter == 2) & (train.building_id != 1099)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train[(train.meter == 2) & (train.building_id != 1099)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train[(train.meter == 2) & (train.building_id != 1099)][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('Only Steam Consumption for All buildings except 1099: Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# ### Overall Mean Energy Consumption - All Meters except Steam

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
train[train.meter != 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train[train.meter != 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train[train.meter != 2][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('All meters except Steam : Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# We can see that there is gradual increase in mean consumption with a peak during Sep to Oct and then back to original levels.

# ### Overall Mean Energy Consumption - All Meters - remove outlier Building 1099

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('All Meters - All buildings except 1099 : Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# This looks just the same as previous plot where we filtered out entire steam meters. So we have captured the true end of energy consumption by just leaving out one outlier building - 1099 -  than filtering out all steam meters. 
# 
# Lets see if the huge steam consumption for building 1099 happens in a specific time period.
# 
# 

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
train[train.building_id == 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(label='By hour', alpha=0.8, title = 'Energy consumption for Building 1099').set_ylabel('Meter reading', fontsize=14);
train[train.building_id == 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(label='By Day', alpha=0.8, title = 'Energy consumption for Building 1099').set_ylabel('Meter reading', fontsize=14);


# There is a huge spike in energy consumption for building 1099 between march and June mid. It drops to zero in June mid till Nov first week and again peaks briefly for just a week and comes to zero back again. 
# 
# It would be interesting to see when building 1099 was constructed.

# In[ ]:


print('Year Built for Building 1099 is :',building_metadata[building_metadata.building_id == 1099].year_built.values)


# In[ ]:


print('Floor count for Building 1099 is :',building_metadata[building_metadata.building_id == 1099].floor_count.values)


# Since the year was not captured, may have been a pretty old building which must be using steam energy.

# We can now revisit earlier plots by removing building 1099 and deduce the true energy consumption trend and compare against what it was when including Outlier

# # Review Plots post outlier 

# ## 1.1A Energy Consumption by Primary Use - Full Data

# In[ ]:


plt.subplot(1,3,2)
train.groupby('primary_use')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings by Primary use')
plt.subplot(1,3,1)
train.groupby('primary_use')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption by Primary use')
plt.subplot(1,3,3)
train.groupby('primary_use')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')
plt.tight_layout()


# ## 1.1B Energy Consumption by Primary Use - Remove Outlier Building 1099

# In[ ]:


plt.subplot(1,3,2)
train[train.building_id != 1099].groupby('primary_use')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings by Primary use')
plt.subplot(1,3,1)
train[train.building_id != 1099].groupby('primary_use')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption by Primary use')
plt.subplot(1,3,3)
train[train.building_id != 1099].groupby('primary_use')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')
plt.tight_layout()


# Since the outlier building 1099 was of Education Primary Use - removing that pushed Education from 1st to 5th highest energy consumer.Services becomes the highest mean enrergy consumer moving from 2nd to 1st position.

# ## 1.2A Energy Consumption by Primary Use by Meter Type - Original Data

# In[ ]:


#train.groupby(['primary_use','meter'])['meter_reading'].mean().sort_values(ascending = False).reset_index().plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')
pivot_df = train.groupby(['primary_use','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()
pivot_df.head()
pivot_df = pivot_df.pivot(index='primary_use', columns='meter_name', values='meter_reading')
pivot_df['AllMeters'] = pivot_df.sum(axis = 1)
pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Hourly Energy consumption by Primary use by meter', stacked = True)
plt.tight_layout()


# ## 1.2B Energy Consumption by Primary Use by Meter Type - Outlier Building 1099 removed

# In[ ]:


#train.groupby(['primary_use','meter'])['meter_reading'].mean().sort_values(ascending = False).reset_index().plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption by Primary use')
pivot_df = train[train.building_id != 1099].groupby(['primary_use','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()
pivot_df.head()
pivot_df = pivot_df.pivot(index='primary_use', columns='meter_name', values='meter_reading')
pivot_df['AllMeters'] = pivot_df.sum(axis = 1)
pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'No 1099: Mean Hourly Energy consumption by Primary use by meter', stacked = True)
plt.tight_layout()


# Same trend as in previous plot. Note types of buildings with Steam as energy tend to consume more energy.

# ## 2.1A Analyze Energy Consumption based on Meter used - With Original Data

# In[ ]:


#plt.figure(figsize=(30,10))
plt.subplot(1,3,2)
train.groupby('meter_name')['building_id'].count().sort_values(ascending = False).plot(kind = 'bar', figsize = (15,6), color = 'purple', title = 'Count of Meter Type')
plt.subplot(1,3,1)
train.groupby('meter_name')['meter_reading'].sum().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Total Energy consumption by Meter Type' )
plt.subplot(1,3,3)
train.groupby('meter_name')['meter_reading'].mean().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Mean Energy consumption by Meter Type' )
plt.tight_layout()


# ## 2.1B Analyze Energy Consumption based on Meter used - Outlier Building 1099 removed

# In[ ]:


#plt.figure(figsize=(30,10))
plt.subplot(1,3,2)
train[train.building_id != 1099].groupby('meter_name')['building_id'].count().sort_values(ascending = False).plot(kind = 'bar', figsize = (15,6), color = 'purple', title = 'Count of Meter Type')
plt.subplot(1,3,1)
train[train.building_id != 1099].groupby('meter_name')['meter_reading'].sum().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Total Energy consumption by Meter Type' )
plt.subplot(1,3,3)
train[train.building_id != 1099].groupby('meter_name')['meter_reading'].mean().sort_values(ascending= False).plot(kind = 'bar', figsize = (15,6), color = 'green', title = 'Mean Energy consumption by Meter Type' )
plt.tight_layout()


# No change in trend. Steam energy due to its scale exceeds other meter types even after removing building 1099. This is due to huge steam consumption by service type buildings. But note that chilled water is catching up with steam once the outlier was removed.

# # Analyze Time Series Energy consumption
# ## 3.1A Analyze mean energy consumption by time - by hours, days and months - Original Data

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# ## 3.1B Analyze mean energy consumption by time - by hours, days and months - Outlier Building 1099 removed

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# Removal of the outlier gave a clear continuous time series trend. Consumption falls from Jan to June continuously before peaking in Sep Oct and then returning to Jan levels.

# ## 3.2A Analyze mean energy consumption by time - by hours, days and months by 'meter type' - Original Data

# In[ ]:


meter_dict = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
# Plot energy consumption by meter
plt.figure(figsize = (20,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Hour')
    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Day')
    train[train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Month')
    plt.legend()
    plt.tight_layout()
plt.show()


# ## 3.2B Analyze mean energy consumption by time - by hours, days and months by **'meter type'** - Outlier Building 1099 removed

# In[ ]:


meter_dict = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
# Plot energy consumption by meter
plt.figure(figsize = (20,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    train[train.building_id != 1099][train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Hour')
    train[train.building_id != 1099][train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Day')
    train[train.building_id != 1099][train.meter == i][['timestamp','meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(title = meter_dict[i], label = 'By Month')
    #plt.title('No 1099: Energy consumption by time by meter type')
    plt.legend()
    plt.tight_layout()
plt.show()


# So once building 1099 was removed, the steam energy trend follows a more continuous pattern starting high at Jan, reducing gradually to a low between June to Sep and climbing up again back to jan levels by the end of the year. This is also the same trend for hotwater usage - kinda nice given steam and hotwater are close cousins:)

# ## 3.3A Analyze Hourly Mean Energy Consumption by Site Id - Original Data

# In[ ]:


plt.subplot(1,3,1)
train.groupby('site_id')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption in sites 0 to 15')
plt.subplot(1,3,2)
train.groupby('site_id')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings in sites 0 to 15')
plt.subplot(1,3,3)
train.groupby('site_id')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption in sites 0 to 15')


# ## 3.3B Analyze Hourly Mean Energy Consumption by Site Id - Outlier Building 1099 removed

# In[ ]:


plt.subplot(1,3,1)
train[train.building_id != 1099].groupby('site_id')['meter_reading'].sum().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Total Energy consumption in sites 0 to 15')
plt.subplot(1,3,2)
train[train.building_id != 1099].groupby('site_id')['building_id'].nunique().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'purple', title = 'Count of Buildings in sites 0 to 15')
plt.subplot(1,3,3)
train[train.building_id != 1099].groupby('site_id')['meter_reading'].mean().sort_values(ascending = False).plot(kind = 'bar', figsize = (20,6), color = 'green', title = 'Mean Energy consumption in sites 0 to 15')


# ## 3.4A Check Time Series Hourly and Daily energy consumption by Site Id - Original Data

# In[ ]:


# Plot energy consumption by site
plt.figure(figsize = (20,20))
for i in range(16):
    plt.subplot(8,2,i+1)
    train[train.site_id == i][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = 'Site {}'.format(i), label = 'By Hour')
    train[train.site_id == i][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(title = 'Site {}'.format(i), label = 'By Day')
    plt.legend()
    plt.tight_layout() # Add tight layout in loop to prevent overlapping text
plt.show()


# ## 3.4B Check Time Series Hourly and Daily energy consumption by Site Id - Remove outlier building 1099

# Check Building 1099 belongs to which site

# In[ ]:


building_metadata[building_metadata.building_id == 1099].site_id.unique()


# It is site 13. Lets check the consumption trend only for Site 13 after removing building 1099.

# In[ ]:


plt.figure(figsize = (20,6))
train[(train.building_id != 1099) & (train.site_id == 13)][['timestamp','meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(title = 'Site 13'.format(i), label = 'By Hour')
train[(train.building_id != 1099) & (train.site_id == 13)][['timestamp','meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(label = 'By Day')


# Site 13 trend clearly changed meaning that the steam meter did not dominate after removing 1099 and other buildings have only other meter types.

# ## 3.5A Energy consumption by Site by meter type - Original Data

# In[ ]:


pivot_df = train.groupby(['site_id','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()
pivot_df.head()
pivot_df = pivot_df.pivot(index='site_id', columns='meter_name', values='meter_reading')
pivot_df['AllMeters'] = pivot_df.sum(axis = 1)
pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)
plt.tight_layout()


## Show values in bar plot
#ax = pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)
#for p in ax.patches:
 #   ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#plt.tight_layout()    


# ## 3.5B Energy consumption by Site by meter type - Outlier Building 1099 removed

# In[ ]:


pivot_df = train[train.building_id != 1099].groupby(['site_id','meter_name'])['meter_reading'].mean().sort_values(ascending = False).reset_index()
pivot_df.head()
pivot_df = pivot_df.pivot(index='site_id', columns='meter_name', values='meter_reading')
pivot_df['AllMeters'] = pivot_df.sum(axis = 1)
pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)
plt.tight_layout()


## Show values in bar plot
#ax = pivot_df.sort_values('AllMeters', ascending = False).drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)
#for p in ax.patches:
 #   ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
#plt.tight_layout()    


# Site 7 and Site 13 exchanged places as removed building 1099 belonged to site 13. This gives a much more concrete picture of which meters are used in which sites. After Steam chilled water plays a major role in large energy consumption.

# # 4. Analyze Building data features

# In[ ]:


building_metadata.head()


# We so far analyzed energy consumption based on buildings, sites, primary use of buildings, type of meter used and across the time scale.
# 
# We have a few more building attributes to analyze
# 
# 1. **square_feet** - Gross Area occupied by the building, more the area more the energy usage. This includes all of the floors of the building.
# 2. **year_built **- year in which building was built. energy is started to get consumed by the building only after the year_built for a building. 
# 3. **floor_count** - Number of floors in the building. more the number of floors more the energy consumption.
# 
# **A note on square_feet and floor_count :** Given that square_feet also includes the area for the total number of floors, total number of floors may perhaps not give additional information to the model. We might choose to ignore the floor count for building a prediction model.If we were to engineer a feature called square_feet / floor, it still cannot predict without also having information on number of floors.

# Before we analyze these 3 features, lets check for missing values for these features.

# In[ ]:


# Define function to report the missing data in a data frame
def report_missing_data(df):
    print('Total Number of rows :', len(df))
    for column in df.columns:
        print(column,':', 'Missing rows:', sum(df[column].isnull()), '|', '% Missing: {:.2f}'.format(sum(df[column].isnull())*100/len(df)),'%')

        # Report missing data for building data
report_missing_data(building_metadata)


# We have square_feet intact with no missing values while floor count is missing in 75% of buildings which is a huge gap. But as mentioned above, if we skip floor_count feature, we do not need to worry about the missing values as well. Since square feet does not have missing data, we are good with it for modeling.
# 
# The year_built feature has 50% missing values. we may not use this feature directly as it is not clear how it will inform the model. Moreover there is no easy way to impute the year_built values so it is quite hard to be able to use this feature at all for prediction. If at all, we plan to use this feature, it will be based on how many years have elapsed since year_built. We will explore this down below at some point.
# 
# Anyway, lets visualize these features to get a quick perspective of these features by itself.

# ## 4.1 Check the distribution of square_feet area for each primary_use

# In[ ]:


plt.figure(figsize = (100,100))
g = sns.FacetGrid(building_metadata, col = 'primary_use',col_wrap = 4)
g.map(sns.distplot,'square_feet', kde = False, label = 'primary_use')
g.set_xticklabels(rotation=90) # rotate all x-axis ticks for all facet subplots
for ax in g.axes.flat:
    plt.setp(ax.get_xticklabels(), visible=True) # Show x axis ticklabels for each facet sub plot
    plt.setp(ax.get_yticklabels(), visible=True) # Show y axis ticklabels for each facet sub plot
plt.tight_layout()


# The square_feet values are all skewed towards right which means more values at the lower ranges of square_feet. We might want to log transform these before we fit linear regression models.

# ## 4.2 Check the distribution of floor_count area for each primary_use

# In[ ]:


plt.figure(figsize = (100,100))
g = sns.FacetGrid(building_metadata, col = 'primary_use',col_wrap = 4)
g.map(sns.distplot,'floor_count', kde = False, label = 'primary_use')
g.set_xticklabels(rotation=90) # rotate all x-axis ticks for all facet subplots
for ax in g.axes.flat:
    plt.setp(ax.get_xticklabels(), visible=True) # Show x axis ticklabels for each facet sub plot
    plt.setp(ax.get_yticklabels(), visible=True) # Show y axis ticklabels for each facet sub plot
plt.tight_layout()


# We do not have floor counts for Services, Food Sales and Service and Religious worship. Also Education, Lodging, office, entertainment, public services have more range of values for floor counts. At this moment, we are not considering this as a valuable feature for prediction.

# ## 4.3 Analyze Year in which buildings were built
# We can visualize trends around year_built in multiple ways to see if we can consider year_built as a valuable feature.
# 
# 1. Trend of count of buildings by year - Increases as years progress
# 2. Trend of energy consumption based on elapsed years - maybe more the elapsed years, more the energy consumed by buildings
# 3. Trend of meter usage by years - number of each types of meters as years progress

# ### 4.3.1 Trend of count of buildings by year - Increases as years progress

# In[ ]:


plt.figure(figsize = (25,6))
train.groupby('year_built')['building_id'].nunique().plot(kind = 'bar', color = 'green', title = 'Number of Buildings built by Year', rot = 90)


# The trend has been more cyclic with too many buildings built in year 1976 as compared to other years. However as we saw considering we have 50% missing values, we do not have sufficient info to conclude anything.

# ### 4.3.2 Trend of energy consumption based on elapsed years

# In[ ]:


# Plot energy consumption by elapsed years
train.groupby('building_id')['elapsed_years','meter_reading'].mean().sort_values('elapsed_years').plot(x = 'elapsed_years',y = 'meter_reading', figsize = (20,6))


# No clear trend that elapsed years of buildings are correlated with energy consumption in a meaningful way. We can can ignore this feature for building  models.

# ### 4.3.3 Trend of meter usage by years - number of each types of meters as years progress

# In[ ]:


pivot_df = train.groupby(['year_built','meter_name'])['meter_reading'].mean().reset_index()
pivot_df.head()
pivot_df = pivot_df.pivot(index='year_built', columns='meter_name', values='meter_reading')
pivot_df.plot(kind = 'bar', figsize = (20,5), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)

#pivot_df['AllMeters'] = pivot_df.sum(axis = 1)
#pivot_df.drop('AllMeters',axis = 1).plot(kind = 'bar', figsize = (20,10), colormap = 'Set1', title = 'Mean Energy consumption by Site Id by meter', stacked = True)
plt.tight_layout()


# Steam and chilled water pretty much dominate in all the years and chilled water proportion has been growing considerably over the last 20 years edging out steam in atleast last 15 years.
# 
# It would be interesting to look at this trend against the square feet area of each building

# ### 4.3.4 Trend of meter usage by building square feet

# In[ ]:


train[['year_built', 'square_feet']].groupby('year_built').mean().plot(kind = 'bar',figsize = (20,5))


# It can be seen that the spikes in mean energy consumption in previous plot is not necessarily due to square_feet area as much as it is due to meter type. We can compare plots 4.3.3 and 4.3.4 for below years:
# 
# 1. 1922 - Sq.ft is very high but energy consumption is pretty low - due to electricity meter being used almost fully.
# 2. 1930 - lower Sq.Ft but higher higher energy consumption as steam being used 40% roughly.
# 3. 1981 - higher energy for lower sq.ft due to chilled water usage.
# 
# For other years like 1938, 1952, 1979 etc. sq.ft correlates with energy consumption only because steam or chilled water is used.
# 
# The importance of meter type type is probably more than sq.ft feature.

# In conclusion for year_built trend, key takeaway is that chilled water usage is more in last 20 years. year_built by itself does not seem to be a very critical feature. Square feet is correlated, but is overshadowed by meter type. 

# # 5. Analyze Weather Features for Train set

# In[ ]:


train.head()


# In[ ]:


print('col index for air_temperature is :', train.columns.get_loc('air_temperature'))
print('col index for wind_speed is :', train.columns.get_loc('wind_speed'))


# In[ ]:


train.iloc[:,9:16].info()


# We have 7 features from air_temperature to wind_speed. All of these are at site level and hence will impact buildings based on which site they are located in. Also all seem to be numeric.
# 
# Before we dive into these features, let us remind ourselves how 1449 buildings are allocated to the 16 sites.

# In[ ]:


sns.set_style('whitegrid')
site_building = building_metadata.groupby('site_id')['building_id'].nunique().sort_values(ascending=False).plot(kind = 'bar', figsize = (10,3), color = 'skyblue')


# Site 3 has the highest number of buildings twice that of the second highest site which is 13. 
# 
# 75% of the sites have atleast 50 buildings each.

# ### Understand Weather data summary stats

# In[ ]:


# Understand summary stats
weather_train.describe()


# ## 5.1 Weather Features - Univariate Plots

# #### Weather Features - Histograms

# In[ ]:


# Select only weather data columns from final train set
columns = train.columns[9:16]
print(columns)

# Create a function to draw histograms for weather columns
def plot_columns(df,columns,plot_type = 'hist'):
    plt.figure(figsize=(20,6))
    for i,column in enumerate(columns):
        plt.subplot(3,3,i+1)
        df[column].plot(kind = plot_type, label = column)
        plt.legend(frameon = False)
    plt.tight_layout()    
    plt.show() 

plot_columns(train,columns)  


# #### Weather Features - Box Plots

# In[ ]:


# Select only weather data columns from final train set
columns = train.columns[9:16]
print(columns)

# Create a function to draw histograms for weather columns
def plot_columns(df,columns,plot_type = 'hist'):
    plt.figure(figsize=(20,6))
    for i,column in enumerate(columns):
        plt.subplot(3,3,i+1)
        sns.boxplot(x = df[column])
        plt.legend(frameon = False)
    plt.tight_layout()    
    plt.show() 

plot_columns(train,columns)


# #### Cloud_Coverage

# In[ ]:


np.sort(train.iloc[:,9:16].cloud_coverage.unique())


# We can see that cloud coverage feature seems to be categorical and not float as shown in data. https://en.wikipedia.org/wiki/Okta - This link shows that cloud coverage is measured from 0 to 9 levels.So we need to convert cloud coverage to categorical. Also we can see that 75% of the values are between 0 and 4.

# #### Precip_depth_1_hr
# Now let us look at the feature : **precip_depth_1_hr** - From the .describe() stats, its clear that about 75% of the values are 0 and hence the single bar in the histogram. Also the min value is -1. Lets do a log transform to see the actual shape of the histogram

# In[ ]:


np.log(train.precip_depth_1_hr+1.001).plot(kind = 'hist') # usually we add 1 to number to be log transformed to ensure input to log is positive. here since w have -1 as min value, we need to add more than 1.


# Inspite of log transform we see a very high skew especially with a value of 0. With this kind of data, this feature may not inform the model much.We might ignore this feature while building models.

# #### Wind Speed
# We can see that wind direction is having a uniform distribution. I guess we can just use this feature as is. 

# #### Other weather features:
# As regards other weather features - **Air Temp, Dew Temp,Sea level pressure and wind speed**, they have minor skews and we can use the **log transform** of these features for building models.

# ## 5.2 Correlation of weather features with mean energy consumption
# We can look for strength of relationship between the weather features and mean energy consumption by using a heat map. We can check correlation at different aggregations of data as this is time series data.

# We cannot determine correlation across the entire data set in a time series data as there will be upward and downward movements in different time periods. It makes sense to break the overall time scale into 3-4 buckets where there is a consitent trend and check correlations only within that bucket so determine if weather features have strong impact on energy consumption.
# 
# Let us review the earlier energy consumption plotted against time by different intervals, hourly, daily and monthly - **with outlier building 1099 removed to show the true trend.**

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15, 6), dpi=100)
train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train[train.building_id != 1099][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# Considering the **energy consumption pattern** is having a distinct pattern in different time periods of the year, We can split the time series data into 4 chunks - 
# 
# 1. **Jan to April - Downward trend**
# 2. **May to Aug - Upward trend**
# 3. **Sep to Oct - Downward trend**
# 4. **Nov to Dec - slight upward trend**
# 
# We will also visualize the weather parameters in these time periods so that we can connect them to energy consumption trend across the year.

# ### Weather parameters over time in the same period

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(15, 6), dpi=100)
train[train.building_id != 1099][['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[0,0], label='By hour', alpha=0.8)#.set_ylabel('Air Temperature', fontsize=14);
train[train.building_id != 1099][['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[0,0], label='By day', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);
train[train.building_id != 1099][['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[0,0], label='By Month', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);
axes[0,0].set_title('Air Temperature by hour, day and month', fontsize=16);
axes[0,0].legend();

train[train.building_id != 1099][['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[0,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train[train.building_id != 1099][['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[0,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train[train.building_id != 1099][['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[0,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[0,1].set_title('Dew Temperature by hour, day and month', fontsize=16);
axes[0,1].legend();

train[train.building_id != 1099][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train[train.building_id != 1099][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train[train.building_id != 1099][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[1,0].set_title('Sea level pressure by hour, day and month', fontsize=16);
axes[1,0].legend();

train[train.building_id != 1099][['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[1,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train[train.building_id != 1099][['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[1,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train[train.building_id != 1099][['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[1,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[1,1].set_title('Wind Speed by hour, day and month', fontsize=16);
axes[1,1].legend();

plt.tight_layout()


# We can see that Air temperature and Dew temperature are highly positively correlated. Sea level pressure and wind speed are in the opposite direction of Air temperature and dew temperature  and hence negatively correlated though the variation is in a small range around the mean unlike air temperature and dew temperature. 
# 
# So we can consider one of air temperature or dew temperature to be more important features than sea level pressure and wind speed.

# ### Break Time Series into 4 buckets 

# In[ ]:


# Split whole time series data into 4 different periods of the year with a consistent energy consumption trend
train_jan_to_apr = train[train.building_id != 1099][train[train.building_id != 1099].month.isin([1,2,3,4])]
train_may_to_aug = train[train.building_id != 1099][train[train.building_id != 1099].month.isin([5,6,7,8])]
train_sep_to_oct = train[train.building_id != 1099][train[train.building_id != 1099].month.isin([9,10])]
train_nov_to_dec = train[train.building_id != 1099][train[train.building_id != 1099].month.isin([11,12])]


# Now we can draw the heatmap correlation plot within each of these periods to see if the weather in the period is correlated with the energy trend in that period.

# ### 5.2.1A Jan to Apr - Energy Consumption Trend - Decreasing

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15,3), dpi=100)
train_jan_to_apr[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train_jan_to_apr[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train_jan_to_apr[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# Energy consumption Correlation with Hourly Data: This captures hourly variations in weather which is probably realistic. But there is too much noise in hourly energy consumption that we do not see any sustained correlation. 
# 
# Energy consumption Correlation with Daily Data: Daily frequency smoothens the noise in measurements of weather as well as meter readings. 
# 
# We will review the weather pattern and energy consumption based on **daily averaged values**.

# ### 5.2.1B Jan to Apr - Correlation Heat Map for Energy usage and weather features

# In[ ]:


# Plot a heatmap between mean daily consumption and the weather features for the period Jan to Apr

# Create Correlation matrix data frame to be plotted as heatmap - subsetting data for Jan to Apr
train_jan_to_apr_corr_day = train_jan_to_apr.groupby('day')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()
train_jan_to_apr_corr_hour = train_jan_to_apr[['hour','meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ]].corr()
train_jan_to_apr_corr_month = train_jan_to_apr.groupby('month')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()

# Create heatmap
plt.figure(figsize=(20,6))
#mask = np.zeros_like(train_corr)
#mask[np.tril_indices_from(mask)] = True
plt.subplot(1,3,1, title = 'Jan to Apr Hourly energy consumption vs Weather') # Add titles for individual subplots
sns.heatmap(train_jan_to_apr_corr_hour,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
plt.subplot(1,3,2, title = 'Jan to Apr Daily energy consumption vs Weather')
sns.heatmap(train_jan_to_apr_corr_day,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
#plt.subplot(1,3,3, title = 'Jan to Apr Monthly energy consumption vs Weather')
#sns.heatmap(train_jan_to_apr_corr_month,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
plt.tight_layout()


# Energy consumption has reasonable negative correlation with Air temperature, Dew temperature and positive correlation with sea level pressure. Also Air/Dew Temperature are positively correlated with each other and negatively correlated with sea level pressure. 

# ### 5.2.1C Jan to Apr - Weather Pattern

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(15, 6), dpi=100)
train_jan_to_apr[['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[0,0], label='By hour', alpha=0.8)#.set_ylabel('Air Temperature', fontsize=14);
train_jan_to_apr[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[0,0], label='By day', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);
train_jan_to_apr[['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[0,0], label='By Month', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);
axes[0,0].set_title('Air Temperature by hour, day and month', fontsize=16);
axes[0,0].legend();

train_jan_to_apr[['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[0,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_jan_to_apr[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[0,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_jan_to_apr[['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[0,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[0,1].set_title('Dew Temperature by hour, day and month', fontsize=16);
axes[0,1].legend();

train_jan_to_apr[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_jan_to_apr[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_jan_to_apr[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[1,0].set_title('Sea level pressure by hour, day and month', fontsize=16);
axes[1,0].legend();

train_jan_to_apr[['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[1,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_jan_to_apr[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[1,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_jan_to_apr[['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[1,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[1,1].set_title('Wind Speed by hour, day and month', fontsize=16);
axes[1,1].legend();

plt.tight_layout()


# We can see that the Air Temp and Dew Temp are rising continuously from Jan to Apr. Sea level pressure dips a little bit. Wind speed remained almost steady with some variation around the mean.
# 
# This is consistent with the observation that the energy usage is negatively correlated with air/dew temp and positively correlated with sea level pressure.

# ### 5.2.2A May to Aug - Energy Consumption Trend - Increasing

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15,3), dpi=100)
train_may_to_aug[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train_may_to_aug[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train_may_to_aug[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# ### 5.2.2B May to Aug - Correlation Heat Map for Enery usage and weather features

# In[ ]:


# Create Correlation matrix data frame to be plotted as heatmap - subsetting data for May to Aug
train_may_to_aug_corr_day = train_may_to_aug.groupby('day')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()
train_may_to_aug_corr_hour = train_may_to_aug[['hour','meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ]].corr()
train_may_to_aug_corr_month = train_may_to_aug.groupby('month')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()

## Create heatmap - May to Aug
plt.figure(figsize=(20,6))
#mask = np.zeros_like(train_corr)
#mask[np.tril_indices_from(mask)] = True
plt.subplot(1,3,1, title = 'May to Aug Hourly energy consumption vs Weather') # Add titles for individual subplots
sns.heatmap(train_may_to_aug_corr_hour,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
plt.subplot(1,3,2, title = 'May to Aug Daily energy consumption vs Weather')
sns.heatmap(train_may_to_aug_corr_day,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
#plt.subplot(1,3,3, title = 'May to Aug Monthly energy consumption vs Weather')
#sns.heatmap(train_may_to_aug_corr_month,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
plt.tight_layout()


# Energy consumption has a positive correlation with air/dew temperature and sea level pressure. Air/Dew Temperature are positively correlated with each other. **wind speed** is **negatively correlated** with **air/dew temperature and sea level pressure**.

# ### 5.2.2C May to Aug - Weather Pattern

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(15, 6), dpi=100)
train_may_to_aug[['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[0,0], label='By hour', alpha=0.8)#.set_ylabel('Air Temperature', fontsize=14);
train_may_to_aug[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[0,0], label='By day', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);
train_may_to_aug[['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[0,0], label='By Month', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);
axes[0,0].set_title('Air Temperature by hour, day and month', fontsize=16);
axes[0,0].legend();

train_may_to_aug[['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[0,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_may_to_aug[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[0,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_may_to_aug[['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[0,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[0,1].set_title('Dew Temperature by hour, day and month', fontsize=16);
axes[0,1].legend();

train_may_to_aug[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_may_to_aug[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_may_to_aug[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[1,0].set_title('Sea level pressure by hour, day and month', fontsize=16);
axes[1,0].legend();

train_may_to_aug[['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[1,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_may_to_aug[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[1,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_may_to_aug[['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[1,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[1,1].set_title('Wind Speed by hour, day and month', fontsize=16);
axes[1,1].legend();

plt.tight_layout()


# We can see that Air/Dew Temp and sea level pressure are increasing throughout while wind speed is decreasing slightly.

# ### 5.2.3A Sep to Oct - Energy consumption trend - Decreasing

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15,3), dpi=100)
train_sep_to_oct[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train_sep_to_oct[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train_sep_to_oct[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# ### 5.2.3B Sep to Oct - Correlation Heat Map for Enery usage and weather features

# In[ ]:


# Create Correlation matrix data frame to be plotted as heatmap - subsetting data for Sep to Oct
train_sep_to_oct_corr_day = train_sep_to_oct.groupby('day')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()
train_sep_to_oct_corr_hour = train_sep_to_oct[['hour','meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ]].corr()
train_sep_to_oct_corr_month = train_sep_to_oct.groupby('month')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()

## Create heatmap - Sep to Oct
plt.figure(figsize=(20,6))
#mask = np.zeros_like(train_corr)
#mask[np.tril_indices_from(mask)] = True
plt.subplot(1,3,1, title = 'Sep to Oct Hourly energy consumption vs Weather') # Add titles for individual subplots
sns.heatmap(train_sep_to_oct_corr_hour,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
plt.subplot(1,3,2, title = 'Sep to Oct Daily energy consumption vs Weather')
sns.heatmap(train_sep_to_oct_corr_day,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
#plt.subplot(1,3,3, title = 'Sep to Oct Monthly energy consumption vs Weather')
#sns.heatmap(train_sep_to_oct_corr_month,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
plt.tight_layout()


# Energy consumption is **negatively correlated** with weather features. Air / Dew Temp are negatively correlated with sea level pressure and wind speed. 

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(15, 6), dpi=100)
train_sep_to_oct[['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[0,0], label='By hour', alpha=0.8)#.set_ylabel('Air Temperature', fontsize=14);
train_sep_to_oct[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[0,0], label='By day', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);
train_sep_to_oct[['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[0,0], label='By Month', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);
axes[0,0].set_title('Air Temperature by hour, day and month', fontsize=16);
axes[0,0].legend();

train_sep_to_oct[['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[0,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_sep_to_oct[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[0,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_sep_to_oct[['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[0,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[0,1].set_title('Dew Temperature by hour, day and month', fontsize=16);
axes[0,1].legend();

train_sep_to_oct[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_sep_to_oct[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_sep_to_oct[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[1,0].set_title('Sea level pressure by hour, day and month', fontsize=16);
axes[1,0].legend();

train_sep_to_oct[['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[1,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_sep_to_oct[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[1,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_sep_to_oct[['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[1,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[1,1].set_title('Wind Speed by hour, day and month', fontsize=16);
axes[1,1].legend();

plt.tight_layout()


# Air / Dew Temp are falling while sea level pressure is slightly increasing. Wind speed remains more or less constant.

# ### 5.2.4A Nov to Dec - Energy consumption trend - Increasing

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15,3), dpi=100)
train_nov_to_dec[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.8).set_ylabel('Meter reading', fontsize=14);
train_nov_to_dec[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=1).set_ylabel('Meter reading', fontsize=14);
train_nov_to_dec[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By Month', alpha=1).set_ylabel('Meter reading', fontsize=14);
axes.set_title('No 1099: Mean Meter reading by hour, day and month', fontsize=16);
axes.legend();


# ### 5.2.4B Nov to Dec - Correlation Heat Map for Enery usage and weather features

# In[ ]:


# Create Correlation matrix data frame to be plotted as heatmap - subsetting data for Nov to Dec
train_nov_to_dec_corr_day = train_nov_to_dec.groupby('day')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()
train_nov_to_dec_corr_hour = train_nov_to_dec[['hour','meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ]].corr()
train_nov_to_dec_corr_month = train_nov_to_dec.groupby('month')['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed' ].mean().reset_index().corr()

## Create heatmap - Nov to Dec
plt.figure(figsize=(20,6))
#mask = np.zeros_like(train_corr)
#mask[np.tril_indices_from(mask)] = True
plt.subplot(1,3,1, title = 'Nov to Dec Hourly energy consumption vs Weather') # Add titles for individual subplots
sns.heatmap(train_nov_to_dec_corr_hour,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
plt.subplot(1,3,2, title = 'Nov to Dec Daily energy consumption vs Weather')
sns.heatmap(train_nov_to_dec_corr_day,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
#plt.subplot(1,3,3, title = 'Nov to Dec Monthly energy consumption vs Weather')
#sns.heatmap(train_nov_to_dec_corr_month,annot=True, cmap=sns.color_palette("BrBG", 7), center = 0) # did not use masking
plt.tight_layout()


# Energy consumption is negatively correlated with Air / Dew Temp which are also  negatively correlated with wind speed. 

# ### 5.2.4C Nov to Dec - Weather Pattern

# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(15, 6), dpi=100)
train_nov_to_dec[['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[0,0], label='By hour', alpha=0.8)#.set_ylabel('Air Temperature', fontsize=14);
train_nov_to_dec[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[0,0], label='By day', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);
train_nov_to_dec[['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[0,0], label='By Month', alpha=1)#.set_ylabel('Air Temperature', fontsize=14);
axes[0,0].set_title('Air Temperature by hour, day and month', fontsize=16);
axes[0,0].legend();

train_nov_to_dec[['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[0,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_nov_to_dec[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[0,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_nov_to_dec[['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[0,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[0,1].set_title('Dew Temperature by hour, day and month', fontsize=16);
axes[0,1].legend();

train_nov_to_dec[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_nov_to_dec[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_nov_to_dec[['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[1,0], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[1,0].set_title('Sea level pressure by hour, day and month', fontsize=16);
axes[1,0].legend();

train_nov_to_dec[['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[1,1], label='By hour', alpha=0.8)#.set_ylabel('dew_temperature', fontsize=14);
train_nov_to_dec[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[1,1], label='By day', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
train_nov_to_dec[['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[1,1], label='By Month', alpha=1)#.set_ylabel('dew_temperature', fontsize=14);
axes[1,1].set_title('Wind Speed by hour, day and month', fontsize=16);
axes[1,1].legend();

plt.tight_layout()


# 
# **In Summary: **
# 
# 1. **Energy usage is in some periods positively correlated and in some periods negatively correlated** with **air / dew temperature**
# 2. **Sea level pressure and Wind speed **have mostly **negative correlation** with **air / dew temperature**
# 3. **Air and Dew Temperature** have significant **positive correlation with each other** and we may even use only one of the two can be used for building a model. 
# 

# ## Weather Pattern across sites

# ### Air Temperature by Site across the year

# In[ ]:


def plot_air_temp_sites(df):
    fig, axes = plt.subplots(8,2, figsize=(15, 15), dpi=100)
    for i in range(16):
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'air_temperature']].set_index('timestamp').resample('H').mean()['air_temperature'].plot(ax=axes[(i%8),(i//8)], label='By hour', alpha=0.8).set_ylabel('Temp in Deg C ', fontsize=10);
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes[(i%8),(i//8)], label='By day', alpha=1).set_ylabel('Temp in Deg C', fontsize=10);
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'air_temperature']].set_index('timestamp').resample('M').mean()['air_temperature'].plot(ax=axes[(i%8),(i//8)], label='By Month', alpha=1).set_ylabel('Temp in Deg C', fontsize=10);
        axes[(i%8),(i//8)].set_title('site {}'.format(i), fontsize=10);
        axes[(i%8),(i//8)].legend(frameon = False, ncol = 3);
    fig.suptitle('Air Temperature by Site across Time', y = 1) # y positions the super title
    plt.tight_layout()  
plot_air_temp_sites(train)


# ### Dew Temperature by Site across the year

# In[ ]:


def plot_dew_temp_sites(df):
    fig, axes = plt.subplots(8,2, figsize=(15, 15), dpi=100)
    for i in range(16):
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'dew_temperature']].set_index('timestamp').resample('H').mean()['dew_temperature'].plot(ax=axes[(i%8),(i//8)], label='By hour', alpha=0.8).set_ylabel('Temp in Deg C ', fontsize=10);
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes[(i%8),(i//8)], label='By day', alpha=1).set_ylabel('Temp in Deg C', fontsize=10);
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'dew_temperature']].set_index('timestamp').resample('M').mean()['dew_temperature'].plot(ax=axes[(i%8),(i//8)], label='By Month', alpha=1).set_ylabel('Temp in Deg C', fontsize=10);
        axes[(i%8),(i//8)].set_title('site {}'.format(i), fontsize=10);
        axes[(i%8),(i//8)].legend(frameon = False, ncol = 3);
    fig.suptitle('Dew Temperature by Site across Time', y = 1) # y positions the super title
    plt.tight_layout()  
plot_dew_temp_sites(train)


# ### Sea Level Pressure pattern across time by Site

# In[ ]:


def plot_sea_pressure_sites(df):
    fig, axes = plt.subplots(8,2, figsize=(15, 15), dpi=100)
    for i in range(16):
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('H').mean()['sea_level_pressure'].plot(ax=axes[(i%8),(i//8)], label='By hour', alpha=0.8).set_ylabel('Pressure', fontsize=10);
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('D').mean()['sea_level_pressure'].plot(ax=axes[(i%8),(i//8)], label='By day', alpha=1).set_ylabel('Pressure', fontsize=10);
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'sea_level_pressure']].set_index('timestamp').resample('M').mean()['sea_level_pressure'].plot(ax=axes[(i%8),(i//8)], label='By Month', alpha=1).set_ylabel('Pressure', fontsize=10);
        axes[(i%8),(i//8)].set_title('site {}'.format(i), fontsize=10);
        axes[(i%8),(i//8)].legend(frameon = False, ncol = 3);
    fig.suptitle('Sea Level Pressure by Site across Time', y = 1) # y positions the super title
    plt.tight_layout()  
plot_sea_pressure_sites(train)


# ### Wind speed pattern across time by Site

# In[ ]:


def plot_wind_speed_sites(df):
    fig, axes = plt.subplots(8,2, figsize=(15, 15), dpi=100)
    for i in range(16):
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'wind_speed']].set_index('timestamp').resample('H').mean()['wind_speed'].plot(ax=axes[(i%8),(i//8)], label='By hour', alpha=0.8).set_ylabel('Meters per sec', fontsize=10);
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes[(i%8),(i//8)], label='By day', alpha=1).set_ylabel('Meters per sec', fontsize=10);
        df[(df.building_id != 1099) & (df.site_id == i)][['timestamp', 'wind_speed']].set_index('timestamp').resample('M').mean()['wind_speed'].plot(ax=axes[(i%8),(i//8)], label='By Month', alpha=1).set_ylabel('Meters per sec', fontsize=10);
        axes[(i%8),(i//8)].set_title('site {}'.format(i), fontsize=10);
        axes[(i%8),(i//8)].legend(frameon = False, ncol = 3);
    fig.suptitle('Wind Speed by Site across Time', y = 1) # y positions the super title
    plt.tight_layout()  
plot_wind_speed_sites(train)


# In[ ]:


# Remove outlier building 1099 from train to use in all data analyses
train_no_outlier = train[train.building_id != 1099]
train_no_outlier.set_index('timestamp', inplace = True)


# ## Energy Consumption by Meter with Daily Weather trend across the year 

# In[ ]:


# Create a dataframe with meter reading, meter type, time stamp and weather parameters.
train_meter_time = train_no_outlier[['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed', 'meter_name']].groupby('meter_name')['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed'].resample('D').mean()
train_meter_time.reset_index(inplace = True)
train_meter_time.head()


# In[ ]:


sns.pairplot(train_meter_time, hue = 'meter_name', markers = ['o','s','D','^'], diag_kws=dict(shade=False))


# Looking at how energy consumption by meter type varied with weather parameters - we can see that 
# 
# 1. As Air/Dew Temperature increased, steam and hotwater consumption came down while chilled water consumption increased. Electricity did not show much of a trend.
# 2. Sea Level Pressure and Wind speed did not have a clear correlation for any of the meters.

# ### Daily Energy consumption and weather features

# In[ ]:


train_no_outlier[['cloud_coverage', 'meter_reading','air_temperature', 'dew_temperature', 'sea_level_pressure','wind_speed']].resample('D').mean().plot(figsize = (20,15), subplots = True)


# No significant correlation between energy usage and cloud coverage

# ### Energy consumption and Weather variation by time of the day and month of the year

# In[ ]:


train_no_outlier[['meter_reading','hour','month', 'air_temperature','dew_temperature', 'sea_level_pressure','wind_speed', 'cloud_coverage']].groupby('hour')['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed'].mean().plot(figsize = (10,10), subplots = True, title = 'By Hour')


# Energy consumption increases during the day peaking at noon and finally coming down by night time. This is as expected. 
# 
# Air temp seems to go down during the day time and increase during the night time which is not making sense. Dew Temp though is behaving as expected - increasing from start to peak at noon and reducing to a low by end of day.
# 
# Sea level pressure and Dew temperature seem to be following a similar pattern within a day 
# 
# wind speed and Air temperature seem to be following a similar pattern within a day 

# In[ ]:


train_no_outlier[['meter_reading','hour','month', 'air_temperature','dew_temperature', 'sea_level_pressure','wind_speed']].groupby('month')['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed'].mean().plot(figsize = (10,10), subplots = True, title = 'By Month')


# ### Analyzing pair plot of energy usage and other weather features by cloud coverage

# In[ ]:


# Create a dataframe with meter reading, meter type, time stamp and weather parameters.
train_meter_cloud = train_no_outlier[['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed', 'cloud_coverage']].groupby('cloud_coverage')['meter_reading','air_temperature','dew_temperature', 'sea_level_pressure','wind_speed'].resample('D').mean()
train_meter_cloud.reset_index(inplace = True)
train_meter_cloud.head()


# In[ ]:


sns.pairplot(train_meter_cloud, hue = 'cloud_coverage')


# No clear pattern between meter reading and cloud coverage

# ### Coming up
# 
# I will start setting up some baseline prediction models and accuracy to get on the leaderboard and start climbing from there.

# Do not hestitate to Upvote if you like :) . Leave a comment for improvement, i will really appreciate and incorporate!
