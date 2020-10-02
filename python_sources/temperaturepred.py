#!/usr/bin/env python
# coding: utf-8

# # Standardized temperature prediction based on history

# In this notebook I'll be using a LSTM NN to predict the temperature based on past observations. For this purpose, the steps followed are:
# 
# 1. Downsampling of the measurements to 1 every 12 hours
# 2. Normalization of both feature and target variables
# 3. Usage of the last 100 entries (120 hours, 5 days) to predict the next 50 entries (60 hours, 2.5 days)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import tensorflow as tf
import statsmodels as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# We will use data from the station 14578001 as training data, and data form the station 22005003 as test data

# In[ ]:


WEATHER_STA = 14578001
TEST_WEATHER_STA = 22005003

def normalize(dataset, target, single_param=False):
    if single_param:
        dataNorm = dataset
        dataNorm[target]=((dataset[target]-dataset[target].min())/(dataset[target].max()-dataset[target].min()))
        return dataNorm
    else:
        dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))
#         dataNorm[target]=dataset[target]
        return dataNorm

def segment(dataset, variable, window = 5000, future = 0):
    data = []
    labels = []
    for i in range(len(dataset)):
        start_index = i
        end_index = i + window
        future_index = i + window + future
        if future_index >= len(dataset):
            break
        data.append(dataset[variable][i:end_index])
        labels.append(dataset[variable][end_index:future_index])
    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out), np.array(true_future), label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out), np.array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


# In[ ]:


df2016 = pd.read_csv(r'/kaggle/input/meteonet/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_2016.csv')
df2017 = pd.read_csv(r'/kaggle/input/meteonet/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_2017.csv')
df2018 = pd.read_csv(r'/kaggle/input/meteonet/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_2018.csv')


# In[ ]:


weather = df2016[(df2016['number_sta'] == WEATHER_STA)]
weather = weather.append(df2017[(df2017['number_sta'] == WEATHER_STA)], ignore_index=True)
weather = weather.append(df2018[(df2018['number_sta'] == WEATHER_STA)], ignore_index=True)
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d %H:%M')
weather.set_index('date', inplace=True)
weather['td'] = weather['td'].interpolate('linear')
weather['precip'] = weather['precip'].interpolate('linear')
weather['hu'] = weather['hu'].interpolate('linear')
weather['ff'] = weather['ff'].interpolate('linear')
weather = weather.drop(['number_sta', 'lat', 'lon', 'height_sta'], axis = 1)

weather_test = df2016[(df2016['number_sta'] == TEST_WEATHER_STA)]
weather_test = weather_test.append(df2017[(df2017['number_sta'] == TEST_WEATHER_STA)], ignore_index=True)
weather_test = weather_test.append(df2018[(df2018['number_sta'] == TEST_WEATHER_STA)], ignore_index=True)
weather_test['date'] = pd.to_datetime(weather_test['date'], format='%Y%m%d %H:%M')
weather_test.set_index('date', inplace=True)
weather_test['td'] = weather_test['td'].interpolate('linear')
weather_test['precip'] = weather_test['precip'].interpolate('linear')
weather_test['hu'] = weather_test['hu'].interpolate('linear')
weather_test['ff'] = weather_test['ff'].interpolate('linear')
weather_test = weather_test.drop(['number_sta', 'lat', 'lon', 'height_sta'], axis = 1)


# In[ ]:


weather = normalize(weather, 'td', single_param=False)
weather_test = normalize(weather_test, 'td', single_param=False)


# In[ ]:


weather_ds = weather.resample('720T').mean()
weather_test_ds = weather_test.resample('720T').mean()

weather_ds = weather_ds.fillna(method='bfill')
weather_test_ds = weather_ds.fillna(method='bfill')


# In[ ]:


HISTORY_LAG = 100
FUTURE_TARGET = 50

X_train, y_train = segment(weather_ds, "td", window = HISTORY_LAG, future = FUTURE_TARGET)
X_train = X_train.reshape(X_train.shape[0], HISTORY_LAG, 1)
y_train = y_train.reshape(y_train.shape[0], FUTURE_TARGET, 1)
print("Data shape: ", X_train.shape)
print("Tags shape: ", y_train.shape)


# In[ ]:


X_test, y_test = segment(weather_test_ds, "td", window = HISTORY_LAG, future = FUTURE_TARGET)
X_test = X_test.reshape(X_test.shape[0], HISTORY_LAG, 1)
y_test = y_test.reshape(y_test.shape[0], FUTURE_TARGET, 1)
print("Data shape: ", X_test.shape)
print("Tags shape: ", y_test.shape)


# In[ ]:


EPOCHS = 200

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(HISTORY_LAG, input_shape=X_train.shape[-2:]),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(FUTURE_TARGET)
])

lstm_model.compile(optimizer='adam', metrics=['mae'], loss='mse')
lstm_model.fit(X_train, y_train, epochs=EPOCHS)


# In[ ]:


yPred = lstm_model.predict(X_test, verbose=0)
y_test = y_test.reshape(y_test.shape[0], FUTURE_TARGET,)


# Now that we have trained the network and used it to predict temperatures using data from the 22005003 station, we can check the results. In the first graph, we well see how would it look if we took every 40th prediction from every 100-long-set 

# In[ ]:


final_list = []
val_final_list = []

for i in yPred:
    final_list.append(i[40])

narray = np.array(final_list)
print(narray.shape)

for i in y_test:
    val_final_list.append(i[40])
    
val_narray = np.array(val_final_list)
print(val_narray.shape)


# In[ ]:


plt.figure(figsize=(30,5))
sns.set(rc={"lines.linewidth": 3})
sns.lineplot(x=np.arange(val_narray.shape[0]), y=val_narray, color="green")
sns.set(rc={"lines.linewidth": 3})
sns.lineplot(x=np.arange(narray.shape[0]), y=narray, color="coral")
plt.margins(x=0, y=0.5)
plt.legend(["Original", "Predicted"])


# Here you can check the result of a random prediction (for example, the 1336th set)

# In[ ]:


multi_step_plot(X_test[1336], y_test[1336], yPred[1336])

