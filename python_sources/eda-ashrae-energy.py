#!/usr/bin/env python
# coding: utf-8

# Started on `16 October 2019`

# # Introduction

# #### This notebook explores the data in the ASHRAE Great Energy Predictor III competition, and make some observations along the way

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime as dt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sns.set()


# In[ ]:


# load data from csv files
train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')
weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')
submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')


# In[ ]:


print('Shape of the data:','\n','  train_csv: ',train.shape,'\n','  test_csv: ',test.shape,'\n',
      '  building_metadata.csv: ',building.shape,'\n','  weather_train.csv: ',weather_train.shape,'\n',
      '  weather_test.csv: ',weather_test.shape)


# There are two main files - `train.csv` and `test.csv`.  In addition to these files, there are also other files, namely:
# * Meta data on the buildings - `building_metadata.csv`
# * Weather information - `weather_train.csv` and `weather_test.csv`

# # Examine the train data

# In[ ]:


# look at the train_data
train.head()


# In[ ]:


# convert 'timestamp' which is a string into datetime object
train['timestamp'] = pd.to_datetime(train['timestamp'])

# let's also extract the pertinent date features like 'hour', 'day', 'day of week' and 'month'
train["hour"] = train["timestamp"].dt.hour.astype(np.int8)
train['day'] = train['timestamp'].dt.day.astype(np.int8)
train["weekday"] = train["timestamp"].dt.weekday.astype(np.int8)
train["month"] = train["timestamp"].dt.month.astype(np.int8)


# In[ ]:


print('Total rows in train: ', len(train))
print('Total buildings: ', train['building_id'].nunique(), '\t', np.sort(train['building_id'].unique()))
print('Number of meter types: ', train['meter'].nunique(), '\t', np.sort(train['meter'].unique()))
print('Total hourly intervals: ', train['timestamp'].nunique(), '\t', np.sort(train['timestamp'].unique()))


# #### Observation:
# * There are 1449 unique building IDs from 0 to 1448.
# * There are 4 types of meters, i.e 0:electricity; 1:chilledwater; 2:steam; 3:hotwater.
# * Timestamp looks to be from 2016-01-01 to 2016-12-31 at hourly intervals, and there is a total of **8784** _(366*24)_ hourly intervals.

# In[ ]:


print('Count of ', train.groupby('meter')['building_id'].nunique())
print('\n', 'Total meters: ', train.groupby('meter')['building_id'].nunique().sum())


# #### Observation:
# * It is stated that not every buildings have all meter types.
# * While the majority of meters are of type '0:electricity', it looks like some buildings don't have electricity meters.
# * There are a total of **2380** meters.
# * Consider over 2016, all these 2380 meters should register a total of 20,905,920 hourly readings _(2380*8784)_. 
# * However, the train data set has **20,216,100** rows. This indicates that there are some missed meter readings, probably.

# ### Visualize how the meter readings varies with timestamp
# * Meter readings were converted to log values for better visualization

# In[ ]:


train['target'] = np.log1p(train['meter_reading'])


# In[ ]:


meter_timestamp = {'h':[], 'd':[], 'w':[], 'm':[]}

for i in range(4):
    x = train.query('meter == @i')
    meter_timestamp['h'].append(x.groupby('hour')['target'].mean())
    meter_timestamp['d'].append(x.groupby('day')['target'].mean())
    meter_timestamp['w'].append(x.groupby('weekday')['target'].mean())
    meter_timestamp['m'].append(x.groupby('month')['target'].mean())


# In[ ]:


plt.figure(figsize=(12,12))
plt.suptitle('Variation of readings for meter types with various time intervals', fontsize=16)
pos = 0
for key,value in meter_timestamp.items():
    plt.subplot(2, 2, pos+1)
    for meter,readings in enumerate(value):
        readings.plot(label=meter)
    pos += 1
    plt.legend()
plt.show()


# #### Observation:
# * There are some significant variations in readings for different meter types with respect to hours, days, days of week and months, in particular months and days of week.
# * Electricity readings are higher in the day and from April to October generally, and lower towards the weekend.
# * Chilled water readings, which is air-conditioning, are also higher in the day, and significantly higher from April to October. The location and sites should be in the northern hemisphere. 
# * Steam and hotwater readings are significant lower in the summer months.
# * We can conclude that the features 'hour', 'day', 'weekday' and 'month' would be very useful for the energy predictions.

# ### Meter Reading (Target) Distribution
# * Let's look at the distribution of the meter readings for each type of the meters. Since the metric is RMSLE, we should plot the log of the meter readings.

# In[ ]:


plt.figure(figsize=(12, 12))
plt.suptitle('Distribution of meter readings by meter type', fontsize=16)
labels = {0:'Electricity', 1:'Chilled Water', 2:'Steam', 3:'Hot Water'}
colors = {0:'navy', 1:'darkorange', 2:'green', 3:'maroon'}
for i in range(4):
    plt.subplot(2,2,i+1)
    meter = train[train['meter']==i]['target']
    g = sns.distplot(meter, color=colors[i])
    plt.xlabel(labels[i])
plt.show()


# * Let's also look at some samples of meter readings for individual buildings.

# ### Electricity meter readings

# In[ ]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

electricity = train[train['meter']==0]
plt.figure(figsize=(20, 40))
for i in range(200):
    plt.subplot(20, 10, i+1)
    bldg = electricity[electricity['building_id']==i]
    g = sns.lineplot(y='meter_reading', x='timestamp', data=bldg, color=colors[0])
    plt.title("building %d" %i,fontsize=10)
    plt.axis('off')
    plt.ylim(0,2000)
plt.show()


# #### Observation:
# * For building_id up to 110+, it looks like there aren't any electricity readings for the first three to four months. These buildings weren't commissioned or occupied during that time period, perhaps. This could be something to think about when data cleansing or when creating ML models.

# ### Chilled water meter readings

# In[ ]:


chilledwater = train[train['meter']==1]
bldg_cw = chilledwater['building_id'].unique()

plt.figure(figsize=(20, 20))
for i in range(100):
    index = bldg_cw[i]
    bldg = chilledwater[chilledwater['building_id']==index]
    plt.subplot(10, 10, i+1)
    g = sns.lineplot(y='meter_reading', x='timestamp', data=bldg, color=colors[1])
    plt.title("building %d" %index,fontsize=10)
    plt.axis('off')
    plt.ylim(0, 6000)
plt.show()


# #### Observation:
# * Chilled water readings look ok for the buildings shown. Generally, there is a trend of higher readings in the summer months.

# ### Steam

# In[ ]:


steam = train[train['meter']==2]
bldg_st = steam['building_id'].unique()

plt.figure(figsize=(20, 20))
for i in range(100):
    index = bldg_st[i]
    bldg = steam[steam['building_id']==index]
    plt.subplot(10, 10, i+1)
    g = sns.lineplot(y='meter_reading', x='timestamp', data=bldg, color=colors[2])
    plt.title("building %d" %index,fontsize=10)
    plt.axis('off')
    plt.ylim(0, 10000)
plt.show()


# ### Hot Water

# In[ ]:


hotwater = train[train['meter']==3]
bldg_hw = hotwater['building_id'].unique()

plt.figure(figsize=(20, 20))
for i in range(100):
    index = bldg_hw[i]
    bldg = hotwater[hotwater['building_id']==index]
    plt.subplot(10, 10, i+1)
    g = sns.lineplot(y='meter_reading', x='timestamp', data=bldg, color=colors[3])
    plt.title("building %d" %index,fontsize=10)
    plt.axis('off')
    plt.ylim(0, 4000)
plt.show()


# #### Observation:
# * Steam and hot water generally has a trough in the summer months.

# # Examine the building meta data

# In[ ]:


building.head()


# In[ ]:


building.describe()


# #### Observation:
# * As stated, 'building_id' is the foreign key to the train and test data. Number of unique IDs matches.
# * Total of **16** site IDs, which is the foreign key to the weather data.

# In[ ]:


print('Total rows in building metadata: ', len(building))
print('Total sites: ', building['site_id'].nunique(), '\t', np.sort(building['site_id'].unique()))
print('Total buildings: ', building['building_id'].nunique(), '\t', np.sort(building['building_id'].unique()))


# In[ ]:


building_site = building.groupby('site_id')['building_id'].count()
plt.figure(figsize=(8,6))
plt.bar(np.arange(16), building_site)
plt.title('Number of buildings by Site', fontsize=16)
plt.xlabel('site_id')
plt.ylabel('Number')
plt.show()


# #### Observation:
# * No discernable pattern here. Furthermore, we do not know the location of the sites.

# In[ ]:


plt.figure(figsize=(12,6))
g = sns.countplot(y='primary_use',data=building)
plt.title('Count of primary uses of the buildings', fontsize=16)
plt.yticks(fontsize=10)
plt.show()


# #### Observation:
# * The majority of the buildings are from education, offices, entertainment/public assembly, and public service and residential.

# In[ ]:


plt.figure(figsize=(12, 6))
g = sns.distplot(building['year_built'].dropna(),bins=24,kde=False)
plt.title('Distribution of buildings by the year they are built', fontsize=16)
plt.yticks(fontsize=10)
plt.show()


# In[ ]:


plt.figure(figsize=(12, 7))
g = sns.scatterplot(y='square_feet',x='floor_count',hue='primary_use',size='square_feet',data=building)
plt.title('XY plot of square feet vs floor count', fontsize=16)
plt.yticks(fontsize=10)
plt.show()


# #### Observation:
# * Looking at the plot, 'square_feet' correspond well with 'floor_count' generally.
# * It is noted that there are many missing values in 'floor_count' (75% missing) and 'year_built' (53% missing). It might not worthwhile to try and impute these missing values.

# # Examine the weather data

# ### Examine weather_train

# In[ ]:


weather_train.head()


# In[ ]:


# convert 'timestamp' which is a string into datetime object
weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])


# In[ ]:


weather_train.describe()


# In[ ]:


print('Total rows in weather_train: ', len(weather_train))
print('Total sites: ', weather_train['site_id'].nunique(), '\t', np.sort(weather_train['site_id'].unique()))
print('Total hourly intervals: ', weather_train['timestamp'].nunique(), np.sort(weather_train['timestamp'].unique()))


# #### Observation:
# * Over 2016 and over the 16 sites, there should register a total of 140,544 hourly readings _(8784*16)_. The weather_train data set has **139,773** rows. This indicates that there are some missed weather observation readings at some sites in the weather_train data.

# ### Examine weather_test

# In[ ]:


weather_test.head()


# In[ ]:


# convert 'timestamp' which is a string into datetime object
weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'])


# In[ ]:


weather_test.describe()


# In[ ]:


print('Total rows in weather_test: ', len(weather_test))
print('Total sites: ', weather_test['site_id'].nunique(), np.sort(weather_test['site_id'].unique()))
print('Total hourly intervals: ', weather_test['timestamp'].nunique(), np.sort(weather_test['timestamp'].unique()))


# #### Observation:
# * Over 2017 to 2018, and over the 16 sites, there should register a total of 280,320 hourly readings.
# * But the weather_test data set has less rows, i.e. **277,243** rows.

# # Finally, let's look at the test data

# In[ ]:


# let's see if the test data correspond to the train data since this is a time series prediction challenge
test.head()


# In[ ]:


# convert 'timestamp' which is a string into datetime object
test['timestamp'] = pd.to_datetime(test['timestamp'])

# let's also extract the pertinent date features like 'hour', 'day', 'day of week' and 'month'
test["hour"] = test["timestamp"].dt.hour.astype(np.int8)
test['day'] = test['timestamp'].dt.day.astype(np.int8)
test["weekday"] = test["timestamp"].dt.weekday.astype(np.int8)
test["month"] = test["timestamp"].dt.month.astype(np.int8)


# In[ ]:


print('Total rows in test: ', len(test))
print('Total buildings: ', test['building_id'].nunique(), '\t', np.sort(test['building_id'].unique()))
print('Number of meter types: ', test['meter'].nunique(), '\t', np.sort(test['meter'].unique()))
print('Total hourly intervals: ', test['timestamp'].nunique(), '\t', np.sort(test['timestamp'].unique()))


# #### Observation:
# * Timestamp looks to be over a two-year period from 2017-01-01 to 2018-12-31 at hourly intervals, a total of **17,500** hourly intervals.

# In[ ]:


print('Count of ', test.groupby('meter')['building_id'].nunique())
print('\n', 'Total meters: ', test.groupby('meter')['building_id'].nunique().sum())


# #### Observation:
# * Similarly, there is a total of **2380** meters.
# * Over the two years from 2017 and 2018, all these 2380 meters should register a total of 41,697,600 hourly readings. The test data set has the same number of **41,697,600** rows. The test data are in order.

# #### That's all for now.....
