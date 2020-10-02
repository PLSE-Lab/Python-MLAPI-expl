#!/usr/bin/env python
# coding: utf-8

# # Welcome to the ASHRAE - Great Energy Predictor Competition
# This notebook is a starter code for all beginners and easy to understand.<br>
# We focus on
# * a simple analysis of the data,
# * create new features,
# * hanlde missing data,
# * encoding and 
# * scale data. <br>
# 
# We use categorical feature encoding techniques, compare<br>
# https://www.kaggle.com/drcapa/categorical-feature-encoding-challenge-xgb
# 
# In this kernel we consider the train data. For prediction we must repeate all operations also for the test data. <br>
# Finally we merge the train data with weahter and building date. After that we define X_train and y_train.

# # Load Libraries
# We need the standard python libraries and some libraries of sklearn.

# In[ ]:


import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import os


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# # Load Data

# In[ ]:


path_in = '../input/ashrae-energy-prediction/'
print(os.listdir(path_in))


# In[ ]:


train_data = pd.read_csv(path_in+'train.csv', parse_dates=['timestamp'])
train_weather = pd.read_csv(path_in+'weather_train.csv', parse_dates=['timestamp'])
building_data = pd.read_csv(path_in+'building_metadata.csv')


# # Help functions

# In[ ]:


def plot_bar(data, name):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    data_label = data[name].value_counts()
    dict_train = dict(zip(data_label.keys(), ((data_label.sort_index())).tolist()))
    names = list(dict_train.keys())
    values = list(dict_train.values())
    plt.bar(names, values)
    ax.set_xticklabels(names, rotation=45)
    plt.grid()
    plt.show()


# # Analysis
# First we do a simple analysis.

# ## Have a look on the data

# In[ ]:


print('# samples train_data:', len(train_data))
print('# samples train_weather:', len(train_weather))
print('# samples building_data:', len(building_data))


# In[ ]:


train_data.head()


# In[ ]:


train_weather.head()


# In[ ]:


building_data.head()


# ## Extract missing data
# * train_data: no missing values
# * train_weather: there are some missing values we have to deal with
# * builing_data: there are missing values for the features year_build and floor_count
# 
# The missing data are numerical values.

# In[ ]:


cols_with_missing_train_data = [col for col in train_data.columns if train_data[col].isnull().any()]
cols_with_missing_train_weather = [col for col in train_weather.columns if train_weather[col].isnull().any()]
cols_with_missing_building = [col for col in building_data.columns if building_data[col].isnull().any()]


# In[ ]:


print(cols_with_missing_train_data)
print(cols_with_missing_train_weather)
print(cols_with_missing_building)


# # Feature engineering
# We handle the missing values, create new features and use encoding techniques based on<br>
# https://www.kaggle.com/drcapa/categorical-feature-encoding-challenge-xgb
# 

# ## Train data
# ### New features
# Based on the timestamp we create new features for the month, the day the hour and the year. These are cyclic features. 

# In[ ]:


train_data['month'] = train_data['timestamp'].dt.month
train_data['day'] = train_data['timestamp'].dt.weekday
train_data['hour'] = train_data['timestamp'].dt.hour


# Additionally we create the feature weekend: 5 = saturday and 6 = sunday.

# In[ ]:


train_data['weekend'] = np.where((train_data['day'] == 5) | (train_data['day'] == 6), 1, 0)


# ### Encoding
# We created the features month, day and hour which are cyclic. 

# In[ ]:


features_cyc = {'month' : 12, 'day' : 7, 'hour' : 24}
for feature in features_cyc.keys():
    train_data[feature+'_sin'] = np.sin((2*np.pi*train_data[feature])/features_cyc[feature])
    train_data[feature+'_cos'] = np.cos((2*np.pi*train_data[feature])/features_cyc[feature])
train_data = train_data.drop(features_cyc.keys(), axis=1)


# There are 4 types of meters: <br>
# 0 = electricity, 1 = chilledwater, 2 = steam, 3 = hotwater <br>
# We use the one hot encoding for this 4 feature.

# In[ ]:


train_data = pd.get_dummies(train_data, columns=['meter'])


# In[ ]:


train_data.head()


# ## Building data
# ### Handle missing data

# In[ ]:


imp_most = SimpleImputer(strategy='most_frequent')


# In[ ]:


building_data[cols_with_missing_building] = imp_most.fit_transform(building_data[cols_with_missing_building])


# ### Encoding
# The feature primary_use is a categorical feature with 16 categories. For the first we use a simple mapping.

# In[ ]:


plot_bar(building_data, 'primary_use')


# In[ ]:


map_use = dict(zip(building_data['primary_use'].value_counts().sort_index().keys(),
                     range(1, len(building_data['primary_use'].value_counts())+1)))


# In[ ]:


building_data['primary_use'] = building_data['primary_use'].replace(map_use)


# In[ ]:


building_data = pd.get_dummies(building_data, columns=['primary_use'])


# ### Scale

# In[ ]:


building_scale = ['square_feet', 'year_built', 'floor_count']


# In[ ]:


mean = building_data[building_scale].mean(axis=0)
building_data[building_scale] = building_data[building_scale].astype('float32')
building_data[building_scale] -= building_data[building_scale].mean(axis=0)
std = building_data[building_scale].std(axis=0)
building_data[building_scale] /= building_data[building_scale].std(axis=0)


# In[ ]:


building_data.head()


# ## Weather data
# 

# In[ ]:


weather_int = ['cloud_coverage']
weather_cyc = ['wind_direction']
weather_scale = ['air_temperature', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_speed']


# ### Handle missing data 

# In[ ]:


imp_most = SimpleImputer(strategy='most_frequent')
train_weather[cols_with_missing_train_weather] = imp_most.fit_transform(train_weather[cols_with_missing_train_weather])


# ### Encoding
# The feature wind_direction is the compass direction (0-360) and cyclic.

# In[ ]:


train_weather['wind_direction'+'_sin'] = np.sin((2*np.pi*train_weather['wind_direction'])/360)
train_weather['wind_direction'+'_cos'] = np.cos((2*np.pi*train_weather['wind_direction'])/360)
train_weather = train_weather.drop(['wind_direction'], axis=1)


# ### Scale

# In[ ]:


mean = train_weather[weather_scale].mean(axis=0)
train_weather[weather_scale] = train_weather[weather_scale].astype('float32')
train_weather[weather_scale] -= train_weather[weather_scale].mean(axis=0)
std = train_weather[weather_scale].std(axis=0)
train_weather[weather_scale] /= train_weather[weather_scale].std(axis=0)


# In[ ]:


train_weather.head()


# # Merge the data

# In[ ]:


train_data = pd.merge(train_data, building_data, on='building_id', right_index=True)
train_data = train_data.sort_values(['timestamp'])
train_data = pd.merge_asof(train_data, train_weather, on='timestamp', by='site_id', right_index=True)
del building_data
del train_weather


# In[ ]:


train_data = train_data.sort_index()


# # Define X_train and y_train

# In[ ]:


no_feature = ['building_id', 'timestamp', 'meter_reading', 'site_id']


# In[ ]:


X_train = train_data[train_data.columns.difference(no_feature)].copy(deep=False)
y_train = train_data['meter_reading']


# In[ ]:


del train_data


# In[ ]:


X_train.head()


# # Scale and rescale y_train
# The target value ist energy consumption in kWh (or equivalent). Note that this is real data with measurement error, which we expect will impose a baseline level of modeling error.
# To train we recommend to scale y_train and rescale the predicted y_test.

# In[ ]:


y_train_scaled = np.log1p(y_train)


# In[ ]:


y_train_scaled.hist(bins=50)


# In[ ]:


y_train_scaled[110:115]


# In[ ]:


y_test = np.expm1(y_train_scaled)


# In[ ]:


y_test[110:115]


# # Evaluation Metric
# The evaluation metric for this competition is Root Mean Squared Logarithmic Error. <br>
# https://www.kaggle.com/c/ashrae-energy-prediction/overview/evaluation

# In[ ]:


def rmse(y_true, y_pred):
    """ root_mean_squared_error """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# # Model
# **Current score: 3.09**
# 
# To create and train a model you can use for example a simple neural network like <br>
# https://www.kaggle.com/drcapa/ashrae-datagenerator-neuralnetwork
# 
# There is also used a DataGenerator. Because of the big data we need a lot of time to train end predict. We recommend to download the kernel and calculate local. So you have more than 9 hours and can reach good results. 

# # Next Steps
# 
# Further the feature enginneering can be extended:
# * Are there any holidays?
# * Is there any realationship between the features?
