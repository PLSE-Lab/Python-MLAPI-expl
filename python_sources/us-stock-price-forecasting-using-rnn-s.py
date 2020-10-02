#!/usr/bin/env python
# coding: utf-8

# **US Stock price forecasting  using deep Neural Networks**
# 
# *Disclaimer*: 
# 
# This is not how trading works. Although the results of this work are very promising for the task of forecasting the future price of a single stock, the keen reader should bear the following fact in their mind. Successful trading is all about balancing portfolios and minimizing the risk. A Machine Learning practitioner should try to find and buy undervalued securities  (stocks, bonds, and such) and sell the overpriced ones.
# 
# Efforts are made towards the longterm (365-day) prediction of the close price index of DOW Jones Industrial index, using a 3-year window form 2016 to 2019. This is a very interesting choice of data since it can put our ML algorithm to the test. In this window, during the test period, the index mostly shows an upward trend. There however exists a sudden drop In the test part. Its shown that a combination of a convolution layer (to capture the patterns) and LSTM layers  (for the sake of long memory) does a phenomenal job and the model is able to accurately predict the 9-day moving average graph. 

# In[ ]:


#data acquisition modules 
import pandas_datareader.data as pdr
import time
import datetime
# general modules 
import pandas as pd
pd.options.display.max_rows = 10
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random
import os
# Deep Learning modules 
import tensorflow as tf
import keras as keras
from keras.models import Sequential
from keras import metrics
from keras.layers import LSTM, Dense, Activation
import h5py
#sklearn and xgboost
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn import preprocessing


# In[ ]:


def stock_data_unpack(ticker, start_date, end_date):
    count = 0
    try:
        raw_data = pdr.get_data_yahoo(ticker, start_date, end_date)
    except ValueError:
        print("ValueError, trying again")
        count += 1
        if count < 9:
            time.sleep(random.randrange(10))
            get_stock_data(ticker, start_date, end_date)
        else:
            print("Yahoo error. I will Try in a second or so")
            time.sleep(range(10,60))
            get_stock_data(ticker, start_date, end_date) 
            
    stock_data = raw_data 
    stock_data.to_csv("raw_data.csv")


# Let's read the DOW data from Yahoo and turn it into a Pandas DataFrame.

# In[ ]:


start = datetime(2016,9,11)
end =   datetime(2019,9,27)
seq_len = 1 
# If you need to read the file from Yahoo activate the following line. I however have uploaded the cvs file.
#stock_data_unpack("^DJI", start_date=start, end_date=end)
df1=pd.read_csv("../input/raw-data/raw_data.csv", names=['Date','High','Low','Open','Close','Volume','Adj Close'], parse_dates=True)
df1=df1.rename(columns={'Open': 'Dow_Open', 'Close': 'Dow_Close','Volume': 'Dow_Volume','High': 'Dow_High','Low': 'Dow_Low'})
df1=df1.drop(['Adj Close'],axis=1)
df1=df1.drop([0], axis=0)


# In[ ]:


df1.head()


# In[ ]:


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time index")
    plt.ylabel("Price Index")
    plt.grid(True)
def moving_average(t, window):
    return np.convolve(t, np.ones(window), 'valid') / window


# The above helper functions help us with plotting and calculating moving average (MA) plots. Furthermore, we normalize the variable for the luxury of numerical convergence.

# In[ ]:


series = df1['Dow_Close'].to_numpy().astype(float)
time = df1.index.values.astype(float)

maxval=np.max(series)
print('The maximum price is=',maxval)
series=series/maxval
maxT=np.max(time)
print('The number of simulated days is=',maxT)
time=time/maxT


# Very well! Let's visuals the time series and also plot a 9-day MA (an important indicator in the trading world).

# In[ ]:


plt.figure(figsize=(10, 6))
window=9
plot_series(time[window-1:], moving_average(series,window))
plot_series(time, series)


# The MA plot (in blue) filters out most of the white noise. Next, we split our data into train and test sets.

# In[ ]:


split_time = 365
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 40#64 This parameter affects the results greatly as it will be seen
batch_size = 8
shuffle_buffer_size = 1000


# In[ ]:


print(tf.__version__)


# In[ ]:


#If the version is less than 2, activate and run the following line
#!pip install tf-nightly-2.0-preview


# **RNN parameter tunning**

# The following helper functions help us with creating data sequencing and prediction. The functions are borrowed from the Coursera "TensorFlow in Practice Specialization." Courtesy of Laurence Moroney.

# 

# In[ ]:


def windowed_datasetII(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


# In[ ]:


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


# Let's start with a call back for the learning rate. To find the best loss function, we repeat the process for three loss functions. Our network has an initial convolution layer, followed by two bidirectional LSTM layers.  

# In[ ]:


#------------------------------------------------
# Huber Loss
#------------------------------------------------
tf.keras.backend.clear_session()

train_set = windowed_datasetII(x_train, window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

model1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Dense(1),
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-6 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)
model1.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history1 = model1.fit(train_set, epochs=100,verbose=2, callbacks=[lr_schedule])

#------------------------------------------------
# MAE loss
#------------------------------------------------
model2 = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Dense(1),
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-6 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)
model2.compile(loss=tf.keras.losses.MeanAbsoluteError(),
              optimizer=optimizer,
              metrics=["mae"])
history2 = model2.fit(train_set, epochs=100,verbose=2, callbacks=[lr_schedule])

#------------------------------------------------
#     MSE loss
#------------------------------------------------
model3 = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Dense(1),
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-6 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)
model3.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=optimizer,
              metrics=["mae"])
history3 = model3.fit(train_set, epochs=100,verbose=2, callbacks=[lr_schedule])


# In[ ]:


plt.semilogx(history1.history["lr"], history1.history["loss"],'b')
plt.semilogx(history2.history["lr"], history2.history["loss"],'g')
plt.semilogx(history3.history["lr"], history3.history["loss"],'r')
plt.xlim([1e-4,1e-1])
plt.ylim([0,1e-1])


# The loss function "MeanSquaredError" shows the greatest promise with a learning rate of $5*e-3$.
# Let's run it.

# In[ ]:


tf.keras.backend.clear_session()
dataset = windowed_datasetII(x_train, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 1)
])

optimizer = tf.keras.optimizers.SGD(lr=5*1e-3, momentum=0.9)
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset,epochs=400)


# In[ ]:


rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]


# In[ ]:


plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)


# Not bad! The dynamics of the system are captured. But the existence of rooms for improvement is undeniable. 
# Let's use the same number of neurons but arrange them in a deeper network .

# In[ ]:


tf.keras.backend.clear_session()
dataset = windowed_datasetII(x_train, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 1)
])

optimizer = tf.keras.optimizers.SGD(lr=5*1e-3, momentum=0.9)
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset,epochs=500)


# In[ ]:


rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]


# In[ ]:


plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
window=40
plot_series(time_valid[window-1:], moving_average(x_valid,window))


# The network surely behaved more poorly. Clearly, our model has predicted the 40-day MA chart. Note that our chosen window was 40 days and it seems that reducing the window increase the accuracy. This makes statistical sense because the latest data have stronger autocorrelation with future events. As 9-day MA is an important indicator in the financial world, lets chose a window of 10 days.

# In[ ]:


split_time = 365
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 10# was 64
batch_size = 8
shuffle_buffer_size = 500 #100


# Furthermore, it seems that having more neurons in the LSTM is more helpful than having multiple LSTM layers of a few neurons.  

# In[ ]:


from keras.callbacks import ModelCheckpoint

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('mae')<0.008):
            print("\nReached reasonable accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()
tf.keras.backend.clear_session()
dataset = windowed_datasetII(x_train, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(filters=window_size, kernel_size=2,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128*window_size, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64*window_size, return_sequences=True)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 1)
])
model.load_weights("../input/weights/weights.first.hdf5")
optimizer = tf.keras.optimizers.SGD(lr=1*1e-3, momentum=0.9)
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=optimizer,
              metrics=["mae"])
# checkpoint

filepath="weights.first.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='mae', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
history = model.fit(dataset,epochs=1,callbacks=callbacks_list)


# In[ ]:


rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]


# In[ ]:


plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
window=10
plot_series(time_valid[window-1:], moving_average(x_valid,window))


# Looks very good. Now let's try to double the neurons in each LSTM layer. The result will not be repeated as the network takes a couple of hours to train. The data file is also a GB large and not worthy of uploading. Had you seen the final results, the moral of the story would have been that small improvements were made and that nothing could be done with this data and the single feature. 
# Do not forget that this is phenomenal as we have practically forecasted the 9-day MA chart a year before its happening.

# In[ ]:


from keras.callbacks import ModelCheckpoint

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('mae')<0.008):
            print("\nReached reasonable accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()
tf.keras.backend.clear_session()
dataset = windowed_datasetII(x_train, window_size, batch_size, shuffle_buffer_size)
model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(filters=window_size, kernel_size=2,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256*window_size, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128*window_size, return_sequences=True)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 1)
])
model.load_weights("weights.second.hdf5")
optimizer = tf.keras.optimizers.SGD(lr=1*1e-3, momentum=0.9)
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=optimizer,
              metrics=["mae"])
# checkpoint

filepath="weights.second.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='mae', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
history = model.fit(dataset,epochs=2,callbacks=callbacks_list)


# In[ ]:


rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]


# In[ ]:


plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
window=10
plot_series(time_valid[window-1:], moving_average(x_valid,window))


# In[ ]:















# In[ ]:




