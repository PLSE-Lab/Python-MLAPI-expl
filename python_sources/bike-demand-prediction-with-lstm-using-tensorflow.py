#!/usr/bin/env python
# coding: utf-8

# ## Bike Demand Prediction with LSTMs using TensorFlow and Keras

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


# ### My goal is to predict the number of future bike shares given the season and weather information.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
from pylab import rcParams
import tensorflow as tf
from tensorflow import keras
sns.set_style("darkgrid")


# In[ ]:


df = pd.read_csv('/kaggle/input/london-bike-sharing-dataset/london_merged.csv', parse_dates=['timestamp'], index_col='timestamp')


# - "timestamp" - timestamp field for grouping the data
# - "cnt" - the count of a new bike shares
# - "t1" - real temperature in C
# - "t2" - temperature in C "feels like"
# - "hum" - humidity in percentage
# - "windspeed" - wind speed in km/h
# - "weathercode" - category of the weather
# - "isholiday" - boolean field - 1 holiday / 0 non holiday
# - "isweekend" - boolean field - 1 if the day is weekend
# - "season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.
# 
# "weathe_code" category description:
# 1 = Clear ; mostly clear but have some values with haze/fog/patches of fog/ fog in vicinity 2 = scattered clouds / few clouds 3 = Broken clouds 4 = Cloudy 7 = Rain/ light Rain shower/ Light rain 10 = rain with thunderstorm 26 = snowfall 94 = Freezing Fog

# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['month'] = df.index.month


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(14, 8))
sns.lineplot(x=df.index, y='cnt',data=df)


# In[ ]:


df_by_month = df.resample('M').sum()
plt.figure(figsize=(14, 8))
sns.lineplot(x=df_by_month.index, y='cnt',data=df_by_month)


# In[ ]:


plt.figure(figsize=(14, 8))
sns.pointplot(x='hour', y='cnt',hue='is_holiday',data=df)


# In[ ]:


plt.figure(figsize=(14, 8))
sns.pointplot(x='day_of_week', y='cnt',data=df)


# In[ ]:


import math
# Get/Compute the number of rows to train the model on
training_data_len = math.ceil(len(df) *.9) # taking 90% of data to train and 10% of data to test
testing_data_len = len(df) - training_data_len

time_steps = 24
train, test = df.iloc[0:training_data_len], df.iloc[(training_data_len-time_steps):len(df)]
print(df.shape, train.shape, test.shape)


# In[ ]:


from sklearn.preprocessing import RobustScaler


# In[ ]:


# Scale the all of the data from columns ['t1', 't2', 'hum', 'wind_speed']
train_trans = train[['t1', 't2', 'hum', 'wind_speed']].to_numpy()
test_trans = test[['t1', 't2', 'hum', 'wind_speed']].to_numpy()
Robust_scale = RobustScaler() # Many outliners exist, so using robustscaler
train.loc[:, ['t1', 't2', 'hum', 'wind_speed']]=Robust_scale.fit_transform(train_trans)
test.loc[:, ['t1', 't2', 'hum', 'wind_speed']]=Robust_scale.fit_transform(test_trans)


# In[ ]:


#Scale the all of the data from columns ['cnt']
train['cnt'] = Robust_scale.fit_transform(train[['cnt']])
test['cnt'] = Robust_scale.fit_transform(test[['cnt']])


# In[ ]:


train.to_numpy()
test.to_numpy()


# In[ ]:


#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(len(train) - time_steps):
    x_train.append(train.drop(columns='cnt').iloc[i:i + time_steps].to_numpy())
    y_train.append(train.loc[:,'cnt'].iloc[i + time_steps])

#Convert x_train and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)


# In[ ]:


#Create the x_test and y_test data sets
x_test = []
y_test = df.loc[:,'cnt'].iloc[training_data_len:len(df)]

for i in range(len(test) - time_steps):
    x_test.append(test.drop(columns='cnt').iloc[i:i + time_steps].to_numpy())
    #y_test.append(test.loc[:,'cnt'].iloc[i + time_steps])

#Convert x_test and y_test to numpy arrays
x_test = np.array(x_test)
y_test = np.array(y_test)


# In[ ]:


# [samples, time_steps, n_features]
# Using all 12 columns of data (take out the bike sharing amount column) to make prediction
print('Train data size:')
print(x_train.shape, y_train.shape)
print('Test data size:')
print(x_test.shape, y_test.shape)


# In[ ]:


#Build the LSTM network model
model = keras.Sequential()
model.add(keras.layers.Bidirectional(
    keras.layers.LSTM(units=50,input_shape=(x_train.shape[1], x_train.shape[2]))))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[ ]:


history = model.fit(x_train, y_train, epochs=200, batch_size=20, validation_split=0.15, shuffle=True)


# In[ ]:


y_pred = model.predict(x_test)
y_pred = Robust_scale.inverse_transform(y_pred)#Undo scaling
y_pred


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_lstm


# In[ ]:


r2 = r2_score(y_test, y_pred)
r2


# In[ ]:


#Pcik some values to zoom in
plt.figure(figsize=(16, 8))
plt.plot(y_test[1200:1600], label='true')
plt.plot(y_pred[1200:1600], label='predicted')
plt.legend()


# In[ ]:





# In[ ]:




