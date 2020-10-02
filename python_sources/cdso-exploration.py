#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, ElasticNet, ElasticNetCV # Lasso is L1, Ridge is L2, ElasticNet is both
from sklearn.model_selection import ShuffleSplit # For cross validation
from sklearn.cluster import KMeans
import lightgbm as lgb # LightGBM is an alternative to XGBoost. I find it to be faster, more accurate and easier to install.

train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
weather_df = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')
meta_df = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')


# ##  Look at what kind of data is in these files

# In[ ]:


train_df.head()


# In[ ]:


weather_df.head()
# There are some missing values. We should also eventually ensure that all of the values fall within a reasonable range. 


# In[ ]:


meta_df.head()
# Missing values as well. 


# ## Explore train_df

# In[ ]:


train_df['meter'].value_counts()


# In[ ]:


train_df['timestamp'][0] 


# ## Timestamp is not in a date time format, so let's convert to the pandas date format, and then add some additional features!

# In[ ]:


train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

train_df['month'] = train_df['timestamp'].dt.month
train_df['weekday'] = train_df['timestamp'].dt.dayofweek
train_df['monthday'] = train_df['timestamp'].dt.day
train_df['hour'] = train_df['timestamp'].dt.hour
train_df['minute'] = train_df['timestamp'].dt.minute


# In[ ]:


train_df['minute'].unique() # Looks like the data doesn't go down to minute resolution. Lets drop it. 


# In[ ]:


train_df = train_df.drop(['minute'], axis = 1)


# ## Look at individual buildings.

# In[ ]:


plt.plot(train_df[train_df['building_id'] == 0]['meter_reading'], alpha = 0.8)
plt.plot(train_df[train_df['building_id'] == 1]['meter_reading'], alpha = 0.8)
plt.plot(train_df[train_df['building_id'] == 2]['meter_reading'], alpha = 0.8)
plt.plot(train_df[train_df['building_id'] == 500]['meter_reading'], alpha = 0.8)


# ### Let's look at the autocorrelation of these plots, and look at a lagplot.

# In[ ]:


pd.plotting.lag_plot(train_df[train_df['building_id'] == 0]['meter_reading'])
plt.plot([0,400],[0,400])
# Look at the 3 clusters. 


# In[ ]:


pd.plotting.lag_plot(train_df[train_df['building_id'] == 500]['meter_reading'])
plt.plot([0,400],[0,400])


# In[ ]:


pd.plotting.autocorrelation_plot(train_df[train_df['building_id'] == 500]['meter_reading'])
plt.show()
pd.plotting.autocorrelation_plot(train_df[train_df['building_id'] == 500]['meter_reading'][:300])
plt.show()


# ### Some buildings have multiple power meters.

# In[ ]:


train_df[train_df['meter'] == 2].head()


# ### The meters are not necessarily consecutive numbers.

# In[ ]:


print(train_df[train_df['building_id'] == 745].meter.unique())
print(train_df[train_df['building_id'] == 1414].meter.unique())


# In[ ]:


train_df[train_df['building_id'] == 745].head()


# ### What does the distribution of weather look like?

# In[ ]:


plt.hist(weather_df['air_temperature'])
plt.show()


# ### We have 3 dataframes, but they should be merged into one so we can feed it to a model.
# ### The site_id will be mapped to the building_id between the train and the meta, and the weather will be mapped to the site and time of the training data.

# In[ ]:


all_df = pd.merge(train_df, meta_df, on = 'building_id', how = 'left')
all_df.head()


# In[ ]:


weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp']) # Convert weather to the correct format before merging
all_df = pd.merge(all_df, weather_df, on = ['site_id', 'timestamp'], how = 'left')
all_df.head()


# ### Now we have the data in a format we can use, so we now need to think of some problems we can solve with this data. It is a good idea to Google what relevant news stories there are around power consumption, buildings, and weather. Then also look to see if there is any relevant research and papers about the topic. 
# 
# ### Some ideas I came up with are:
# ### 1. New York recently came up with a tax on inefficient buildings. How will this tax affect power consumption?
# ### 2. How will climate change affect power consumption?
# ### 3. What is the most effective way to transition to renewable power? What combinaiton of Solar, Wind and Batteries would be the most cost effective and would be robust to prolonged bad weather. 
# 

# 
# ### Then consider some relevant factors such as: How difficult it will be to answer the question, what external data can be brough in, is there any relevant research, how much will the topic impress judges?
# 
# ### 1. We would likely need some electricity/building pricing information or data before and after the tax/policy change to be able to solve it. This is an active topic of discussion in NYC, so there are probably many different opinions that can be discussed in good narrative. There might be difficulties since data is noisy, and there might not be a visible reaction from changes to electricity prices (risky). Maybe there are some other policies which can be better examined, but if this can be pulled off, it will be a top contender. 
# ### 2. This is very do-able, even with just linear regression. Not much external data required, just some estimates of the effects of climate change. Just create a model which predicts power usage from weather and adjust the weather to climate change predictions. Might be less impressive to the judges, but has serious potential to win if done right.
# ### 3.  Solvable, but harder than 2. Would need pricing data on renewable power, and how to convert weather to power generation. Very impressive if you manage to solve it (similar idea won us the Championship).
# 
# ### Let's take a crack at #2 because I am feeling a bit lazy. 

# In[ ]:


data = all_df.groupby(['month', 'monthday'])[['meter']].sum()         .join(all_df.groupby(['month', 'monthday'])['air_temperature'].mean()).reset_index()
data.head()

