#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv('../input/TS.csv')
dataset.Date = pd.to_datetime(dataset.Date)
dataset.set_index('Date')
dataset.info()


# In[ ]:


def plot_dataset(x, y, second_color_index = 0, original_x = None, original_y = None):
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y-%m')
    monthsFmt = mdates.DateFormatter('%m')

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    if (second_color_index == 0):
        ax.plot(x, y)
    else:
        x1, y1 = x[:second_color_index], y[:second_color_index]
        ax.plot(x1, y1, color='blue')
        x2, y2 = x[second_color_index:], y[second_color_index:]
        ax.plot(x2, y2, color='red')
    
    if (original_x is not None):
        ax.plot(original_x, original_y, color='green')
            
    # format the ticks
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(yearsFmt)

    # format the coords message box
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.show()
    
plot_dataset(dataset.Date, dataset.Price)


# In[ ]:


training_set = dataset.Price.values
training_set = training_set.reshape(-1,1)

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
print(len(training_set_scaled))


# In[ ]:


# Creating a data structure with window_size timesteps and t+1 output
window_size = 20
n_features = 1

X_train = []
y_train = []
for i in range(window_size, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-window_size:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape, y_train.shape)

# Reshaping, making inputs ready for RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
print(X_train.shape, y_train.shape)


# In[ ]:


# Create LSTM model
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], n_features)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mse', metrics=['acc'])
history = model.fit(X_train, y_train, epochs = 800, batch_size = 32)


# In[ ]:


# plot history
plt.plot(history.history['loss'], label='train loss')
#plt.plot(history.history['val_loss'], label='test')
plt.plot(history.history['acc'], label='train acc')
#plt.plot(history.history['val_acc'], label='test')
#plt.plot(history.history['mean_squared_error'], label='train mse')
#plt.plot(history.history['val_mean_squared_error'], label='test')
plt.title('Sales Amount Prediction')
plt.xlabel('Time')
plt.ylabel('Sales Amount')
plt.legend()
plt.show()


# ##Making the predictions

# In[ ]:


test_size = 60

inputs = dataset.Price[len(dataset) - test_size - window_size:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(window_size, len(inputs)):
    X_test.append(inputs[i-window_size:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))
predicted_sales_amount = model.predict(X_test)
predicted_sales_amount = sc.inverse_transform(predicted_sales_amount)
#print(predicted_sales_amount.shape)

last_date = dataset.Date.iloc[-1];
dataset_with_predictions = dataset.copy()
for i in range(0,len(predicted_sales_amount)):
    next_date = last_date + timedelta(days=1)
    dataset_with_predictions = dataset_with_predictions.append({'Date':next_date, 'Price':predicted_sales_amount[i][0]}, ignore_index=True)
    last_date = next_date
    
print(len(dataset_with_predictions))    
dataset_with_predictions.tail()


# In[ ]:


prediction_start = len(dataset) - len(predicted_sales_amount)
plot_dataset(dataset_with_predictions.Date, dataset_with_predictions.Price, prediction_start, dataset.Date, dataset.Price)


# In[ ]:


dt = pd.to_datetime('2018-03-01')
original_set = dataset.copy()
original_set.set_index('Date', inplace=True)
original_y = original_set.loc['2018-03-01':]
#original_y.head()

predictde_set = dataset_with_predictions.copy()
predictde_set.set_index('Date', inplace=True)
predicted_y = predictde_set.loc['2018-03-01':'2018-04-30']
#predicted_y.head()

# calculate RMSE for intersection of dates
rmse = np.sqrt(mean_squared_error(original_y.Price, predicted_y.Price))
print('Test RMSE: %.3f' % rmse)

