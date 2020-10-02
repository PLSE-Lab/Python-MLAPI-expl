#!/usr/bin/env python
# coding: utf-8

# # ASHRAE - The first step: EDA with Python
# 
# In this competition we will develop models to predict the energy usage in each building. The dataset contains 1450+ buildings information. Different buildings may have different types of meters.
# 
# Before we start to build a model, we should have some intuition about the data first. So using visualization tools such as Seaborn we can be more familiar with the data we can get.
# 
# If you find this notebook interesting or useful, please do not hesitate to vote for me!
# 
# ## 1 Start from Here
# 
# ### 1.1 Loading Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
import gc
sns.set()

PATH='/kaggle/input/ashrae-energy-prediction/'
train=pd.read_csv(PATH+'train.csv')
test=pd.read_csv(PATH+'test.csv')
weather_train=pd.read_csv(PATH+'weather_train.csv')
weather_test=pd.read_csv(PATH+'weather_test.csv')
building=pd.read_csv(PATH+'building_metadata.csv')


# In[ ]:


# merge the data
train = train.merge(building, on='building_id', how='left')
test = test.merge(building, on='building_id', how='left')
train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
test = test.merge(weather_test, on=['site_id', 'timestamp'], how='left')
del weather_train, weather_test,building
gc.collect()


# ### 1.2 Check some Information about the Data
# 
# Before we start, let's check the missing ratio of each column:

# In[ ]:


# check whether there are missing values
data_na=(train.isnull().sum()/len(train))*100
data_na=data_na.drop(data_na[data_na==0].index).sort_values(ascending=False)
missing_data=pd.DataFrame({'MissingRatio':data_na})
print(missing_data)


# We can see that after merging the dataset some of the columns contain a large percentage of missing values, such as 'floor_count' and 'year_bulit'. We can further check the information of the dataset.

# In[ ]:


train.info()


# We can save the store memory by transforming some of the data types.

# In[ ]:


# Saving the memory space
data_types = {'building_id': np.int16,
          'meter': np.int8,
          'site_id': np.int8,
          'square_feet': np.int32,
          'year_built': np.float16,
          'floor_count': np.float16,    
          'cloud_coverage': np.float16,
          'precip_depth_1_hr': np.float16,
           'wind_direction': np.float16,     
          'dew_temperature': np.float32,
          'air_temperature': np.float32,
          'sea_level_pressure': np.float32,
          'wind_speed': np.float32,
          'primary_use': 'category',}

for feature in data_types:
    train[feature] = train[feature].astype(data_types[feature])
    test[feature] = test[feature].astype(data_types[feature])
    
train["timestamp"] = pd.to_datetime(train["timestamp"])
test["timestamp"] = pd.to_datetime(test["timestamp"])
gc.collect();


# Now let's see the data sample.

# In[ ]:


train.head()


# ## 2 Data Visualization
# 
# ### 2.1 Boxplot
# 
# First let's see the relationship between the usage of the building and the age of the building:

# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(x='primary_use', y='year_built', data=train)
plt.xticks(rotation=90)


# It seems that some kinds of buildings were built eariler, such as those used as technology building. What about the relationship between the building area and building types?

# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(x='primary_use', y='square_feet', data=train)
plt.xticks(rotation=90)


# The above is an intuitive connection, as different types of buildings may install different types of meters, we will the this factor into consideration.

# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(x='primary_use', y='square_feet', hue='meter',data=train)
plt.xticks(rotation=90)


# The result are clear. Some of the buildings only install two or three types of meters, and the corresponding building area may also have significant differences.

# In[ ]:


sample=pd.DataFrame(train,columns=['site_id','primary_use'])
sample.drop_duplicates(keep='first')


# We can see that the buildings with a same 'site_id' may have different primary usage.
# 
# ### 2.2 View the trends of the time series data
# 
# In this competition the data is shown in time series form, each row has a corresponding Time Label. So it is important for us to visualize the trends of the data.
# 
# First of all, we will observe the change of data from a monthly perspective. Specifically, the data will be separated by 'site_id' and the meter types.

# In[ ]:


# We can see it by month
fig, axes = plt.subplots(8,2,figsize=(15, 30))
color_dic={'red':0,'blue':1,'orange':2,'purple':3}
for i in range(0,15):    
    for color,meter in color_dic.items():
        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):
            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by month', fontsize=13)
        axes[i%8][i//8].legend();
        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);
        plt.subplots_adjust(hspace=0.45)


# From the visualization result, we can observe some rules of the data changing. For example, in most buildings the values of meter type 2 and 3 have a significant decrease during the summer, because type 2 provides steam and type 3 provides hot water. But the condition of site_id 13 have a little different.
# 
# We can also observe the data from a weekly perspective:

# In[ ]:


# We can also see it by week
fig, axes = plt.subplots(8,2,figsize=(15, 30))
color_dic={'red':0,'blue':1,'orange':2,'purple':3}
for i in range(0,15):    
    for color,meter in color_dic.items():
        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):
            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('W').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by week', fontsize=13)
        axes[i%8][i//8].legend();
        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);
        plt.subplots_adjust(hspace=0.45)


# From the weekly perspective,  some data show a cyclical pattern such as the condition of site_id 12.
# 
# And what about from a daily perspective?

# In[ ]:


# By day
fig, axes = plt.subplots(8,2,figsize=(15, 30))
color_dic={'red':0,'blue':1,'orange':2,'purple':3}
for i in range(0,15):    
    for color,meter in color_dic.items():
        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):
            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by day', fontsize=13)
        axes[i%8][i//8].legend();
        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);
        plt.subplots_adjust(hspace=0.45)


# We can also further separate the data by hours:

# In[ ]:


# By hour
fig, axes = plt.subplots(8,2,figsize=(15, 30))
color_dic={'red':0,'blue':1,'orange':2,'purple':3}
for i in range(0,15):    
    for color,meter in color_dic.items():
        if(len(train[(train['site_id']==i)&(train['meter']==meter)])!=0):
            train[(train['site_id']==i)&(train['meter']==meter)][['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i%8][i//8], alpha=0.9, label=str(meter), color=color).set_ylabel('Mean meter reading by hour', fontsize=13)
        axes[i%8][i//8].legend();
        axes[i%8][i//8].set_title('site_id {}'.format(i), fontsize=13);
        plt.subplots_adjust(hspace=0.45)


# ### 2.3 The Weather Condition
# 
# The weather condition may also have connections with the energy consumption, we can visualize it, too.

# In[ ]:


# the weather condition(by day)
fig, axes = plt.subplots(figsize=(20,8))
axes1=axes.twinx()
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes1,alpha=0.9, label='meter_reading', color='green')
train[['timestamp', 'air_temperature']].set_index('timestamp').resample('D').mean()['air_temperature'].plot(ax=axes,alpha=0.9, label='air_temperature', color='red')
train[['timestamp', 'dew_temperature']].set_index('timestamp').resample('D').mean()['dew_temperature'].plot(ax=axes,alpha=0.9, label='dew_temperature', color='blue')
plt.legend()


# The green line represents 'meter_reading', we may not find any correlation between the temperature and the meter_reading, and it is strange that the value come across a sudden drop in Jun.
# 
# We can see the cloud condition.

# In[ ]:


# the cloud condition(by day)
fig, axes = plt.subplots(figsize=(20,8))
axes1=axes.twinx()
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes1,alpha=0.9, label='meter_reading', color='green')
train[['timestamp', 'cloud_coverage']].set_index('timestamp').resample('D').mean()['cloud_coverage'].plot(ax=axes,alpha=0.9, label='cloud_coverage', color='cyan')
plt.legend()


# We can see the wind condition, too.

# In[ ]:


# the wind condition(by day)
fig, axes = plt.subplots(figsize=(20,8))
axes1=axes.twinx()
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes1,alpha=0.9, label='meter_reading', color='green')
train[['timestamp', 'wind_speed']].set_index('timestamp').resample('D').mean()['wind_speed'].plot(ax=axes,alpha=0.9, label='wind_speed', color='purple')
plt.legend()


# In the visualization report above, we can see that the weather conditions are almost random, and there seems no correlation between weather conditions and the meter_reading if we just take such a cursory look.
# 
# ### 2.4 Check the correlation matrix
# 
# Using the heatmap in seaborn can benefit us a lot, we can observe the corrlation of various features in this map.

# In[ ]:


for i in range(0,4):
    corr = train[train.meter == i][['timestamp','meter_reading','square_feet','year_built','floor_count',
             'air_temperature','cloud_coverage','dew_temperature','sea_level_pressure','wind_direction','wind_speed']].corr()
    f, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(corr,annot=True,cmap='RdGy')


# Now we have a basic understanding towards the data and the next step will be feature engineering based on the characteristics of the data and we will try a few types of models.
