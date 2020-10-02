#!/usr/bin/env python
# coding: utf-8

# # Intro Bike Sharing Demand Competition
# Source: https://www.kaggle.com/c/bike-sharing-demand
# 
# This notebook is a starter code for all beginners and easy to understand. We will give an introduction to analysis and feature engineering.<br> 
# Therefore we focus on
# * a simple analysis of the data,
# * create new features,
# * encoding and
# * scale data.
# 
# We use categorical feature encoding techniques, compare <br>
# https://www.kaggle.com/drcapa/categorical-feature-encoding-challenge-xgb <br>
# 
# For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month. You must predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.

# # Load Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import calendar


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[ ]:


from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# # Input path

# In[ ]:


path_in = '../input/'
print(os.listdir(path_in))


# # Load Data

# In[ ]:


train_data = pd.read_csv(path_in + 'train.csv', parse_dates = ['datetime'],
                         index_col='datetime', infer_datetime_format=True)
test_data = pd.read_csv(path_in + 'test.csv', parse_dates = ['datetime'],
                        index_col='datetime', infer_datetime_format=True)
samp_subm = pd.read_csv(path_in+'sampleSubmission.csv', parse_dates = ['datetime'],
                        index_col='datetime', infer_datetime_format=True)


# # Functions

# In[ ]:


def plot_bar(data, feature):
    """ Plot distribution """
    
    fig = plt.figure(figsize=(5,3))
    sns.barplot(x=feature, y='count', data=data, palette='Set3',orient='v')


# In[ ]:


def plot_timeseries(data, feature):
    """ Plot timeseries """
    
    fig = plt.figure(figsize=(16,9))
    plt.plot(data.index, data[feature])
    plt.title(feature)
    plt.grid()


# In[ ]:


def plot_timeseries_train_and_predict(train, predict, year, month):
    """ Compare train and predict data for a month """
    
    start_date = datetime.datetime(year, month, 1, 0, 0, 0)
    last_day_of_month = calendar.monthrange(year, month)[1]
    end_date = datetime.datetime(year, month, last_day_of_month, 23, 0, 0)
    
    fig = plt.figure(figsize=(16,9))
    plt.plot(train[start_date: end_date].index, train.loc[start_date:end_date, 'count'], 'b', label = 'train')
    plt.plot(predict[start_date: end_date].index, predict.loc[start_date:end_date, 'count'], 'r', label = 'predict')
    plt.title('Train and Predict')
    plt.legend()
    plt.grid()


# In[ ]:


def rmse(y_true, y_pred):
    """ root_mean_squared_error """
    
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# # EDA
# You are provided hourly rental data spanning two years. For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month. You must predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.

# In[ ]:


# Parameters
num_months_per_year = 12
year_list = [2011, 2012]


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# ## Trend

# In[ ]:


month = 5
year = 2011
start_date = datetime.datetime(year, month, 1, 0, 0, 0)
end_date = datetime.datetime(year, month, 19, 23, 0, 0)
# train_data['count_log'] = np.log1p(train_data['count'])
# train_data['rolling_mean'] = train_data['count'].rolling(window = 24).mean()
# train_data['rolling_std'] = train_data['count'].rolling(window = 24).std()


# # Missing Timestamps
# There are losts of missing hours in the train dataset. We expect $ 2 years*12 months*19 days *24 hours = 10944 timesteps$. We count 10886 timesteps so there are 58 missing. Every month in the train data set hast 456 timestamps.  

# Fill missing timestamps:

# In[ ]:


train_data_temp = pd.DataFrame(columns=train_data.columns)

for year in year_list:
    for month in range(num_months_per_year):
        start_date = datetime.datetime(year, month+1, 1, 0, 0, 0)
        end_date = datetime.datetime(year, month+1, 19, 23, 0, 0)
        # Fill missing timestamps
        temp = train_data[start_date:end_date].resample('H').asfreq()
        # Handle missing values
        features_fill_zero = ['casual', 'registered', 'count']
        temp[features_fill_zero] = temp[features_fill_zero].fillna(0)
        features_fill_bbfil = ['season', 'holiday', 'workingday', 'weather']
        temp[features_fill_bbfil] = temp[features_fill_bbfil].fillna(method='bfill')
        features_fill_linear = ['temp', 'atemp', 'humidity', 'windspeed']
        temp[features_fill_linear] = temp[features_fill_linear].interpolate(method='linear')
        
        train_data_temp = train_data_temp.append(temp)
        
train_data = train_data_temp


# The datetime and the seasons are cyclic features. So we can use a cyclic encoding for it.

# # Create new features
# Based on the datetime we create new features for the month, the weekday the hour and the year. These are also cyclic features.

# In[ ]:


train_data['weekday'] = train_data.index.weekday
train_data['hour'] = train_data.index.hour
test_data['weekday'] = test_data.index.weekday
test_data['hour'] = test_data.index.hour


# In[ ]:


train_data.head()


# ## Feature Season

# In[ ]:


plot_bar(train_data, 'season')
plt.grid()


# ## Feature Weekday

# In[ ]:


plot_bar(train_data, 'weekday')
plt.grid()


# ## Feature Hour
# Add new feature hour_group by group of hours.

# In[ ]:


plot_bar(train_data, 'hour')
plt.grid()


# In[ ]:


def hour_group(s):
    if((0<=s) & (s<=6)):
        return 1
    elif((s==7) | (s==9)):
        return 2
    elif((s==8) | (s==16) | (s==19)):
        return 3
    elif((10<=s) & (s<=15)):
        return 4
    elif((s==17) | (s==18)):
        return 5
    elif(20<=s):
        return 6


# In[ ]:


#train_data['hour_group'] = train_data['hour'].apply(hour_group)
#test_data['hour_group'] = test_data['hour'].apply(hour_group)


# # Encoding 
# ## Cyclic features

# In[ ]:


# features_cyc = ['hour', 'weekday']
# for feature in features_cyc:
#     train_data[feature+'_sin'] = np.sin((2*np.pi*train_data[feature])/max(train_data[feature]))
#     train_data[feature+'_cos'] = np.cos((2*np.pi*train_data[feature])/max(train_data[feature]))
#     test_data[feature+'_sin'] = np.sin((2*np.pi*test_data[feature])/max(test_data[feature]))
#     test_data[feature+'_cos'] = np.cos((2*np.pi*test_data[feature])/max(test_data[feature]))
# train_data = train_data.drop(features_cyc, axis=1)
# test_data = test_data.drop(features_cyc, axis=1)


# ## One Hot Encoding for categorical variables

# In[ ]:


features_one_hot = ['weekday', 'hour_group', 'weather']
train_data[features_one_hot] = train_data[features_one_hot].astype(int).astype(str)
test_data[features_one_hot] = test_data[features_one_hot].astype(int).astype(str)


# In[ ]:


train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)


# # Scale Data

# In[ ]:


scale_features = ['temp', 'atemp', 'humidity', 'hour', 'windspeed']


# In[ ]:


scaler = MinMaxScaler()
train_data[scale_features] = scaler.fit_transform(train_data[scale_features])
test_data[scale_features] = scaler.transform(test_data[scale_features])


# # Predict Monthly

# In[ ]:


# Features
feature_list = ['holiday', 'workingday', 'weather', 'temp', 'atemp',
                'humidity', 'windspeed', 'hour_group',
                'hour', 'weekday']
no_features = ['casual', 'registered', 'count']


# In[ ]:


predictions = []
for year in year_list:
    for month in range(num_months_per_year):
        # Train model
        start_date = datetime.datetime(year, month+1, 1, 0, 0, 0)
        end_date = datetime.datetime(year, month+1, 19, 23, 0, 0)
        X_train = train_data[start_date:end_date][train_data.columns.difference(no_features)].copy()
        y_train = train_data[start_date:end_date]['count'].copy()
        
        y_train = np.log1p(y_train)

        #model = XGBRegressor(n_estimators = 100, random_state=2020)
        model_rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1, max_features='auto')
        model_gbr = GradientBoostingRegressor(n_estimators=1000)
        model_rfr.fit(X_train, y_train)
        model_gbr.fit(X_train, y_train)

        # Predict test data
        start_date = datetime.datetime(year, month+1, 20, 0, 0, 0)
        last_day_of_month = calendar.monthrange(year, month+1)[1]
        end_date = datetime.datetime(year, month+1, last_day_of_month, 23, 0, 0)
        X_test = test_data[start_date:end_date][train_data.columns.difference(no_features)].copy()
        
        y_test_rfr = model_rfr.predict(X_test)
        y_test_gbr = model_rfr.predict(X_test)
        
        y_test = 0.0 * y_test_gbr + 1.0 * y_test_rfr
        y_test = np.expm1(y_test)
        
        predictions.extend(y_test)


# Set negative values to zero:

# In[ ]:


predictions = [0 if i < 0 else i for i in predictions]


# # Generate Output

# In[ ]:


output = pd.DataFrame({'datetime': test_data.index,
                       'count': predictions})
output.to_csv('submission.csv', index=False)


# # Analyse Results

# In[ ]:


predict = pd.DataFrame(index=output['datetime'])
predict['count'] = output['count'].values
predict.head()
plot_timeseries_train_and_predict(train_data, predict, 2011, 2)


# In[ ]:


fig = plt.figure(figsize=(16,9))
plt.plot(train_data.index, train_data['count'], 'b', label = 'train')
plt.plot(output['datetime'],output['count'], 'r', label = 'test')
plt.title('Train and Test')
plt.legend()
plt.grid()

