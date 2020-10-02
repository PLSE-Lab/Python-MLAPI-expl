#!/usr/bin/env python
# coding: utf-8

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


# **Import Libraries**

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# **Open training file**

# In[ ]:


df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


print(df[df["Province_State"]=="New York"])


# **Add new column for time named as days**

# In[ ]:


df["days"]=[x for x in range(1,307) for x in range(1,75)]


# In[ ]:


df.info()


# **Filter New York, for forecasting New York**

# In[ ]:


df=df[df["Province_State"]=="New York"]


# In[ ]:


df.info()


# In[ ]:


df.head()
x=df.iloc[:,1]
time=df.iloc[:,6]
y=df.iloc[:,4]
time=time.to_numpy(dtype="float32")
series=y.to_numpy(dtype="float32")
time.shape


# In[ ]:


plt.figure(figsize=(10, 6))

plt.plot(time, series)
plt.title("Confirmed Cases in New York")
plt.ylabel("Confirmed Cases")
plt.xlabel("Days")


# **Split the training Set into training and Validation. Training set is until 70 days last 4 days will be predicted. The last 4 days will serve to select which method is the best for forecasting.**

# In[ ]:


time=np.array(time)
series=np.array(series)
split_time = 71
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


# In[ ]:


window_size = 2
batch_size = 3
shuffle_buffer_size = 71


# In[ ]:


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  
  return dataset


# In[ ]:


dataset = windowed_dataset(x_train, window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)


# In[ ]:


print(x_train)


# **First Method is single neuron regression**

# In[ ]:


l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])

model.compile(loss=tf.keras.losses.Huber(), optimizer="adam")
model.fit(dataset,epochs=100)


# In[ ]:


forecast=[]
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
  

forecast = forecast[split_time-window_size:]
print(forecast)
results = np.array(forecast)[:, 0, 0]
print(forecast)

plt.figure(figsize=(10, 6))

line1=plt.plot( time_valid, x_valid,label="Real")
line2=plt.plot(time_valid, results,label="Forecasted")
plt.title("New York Single Neuron Forecasting")
plt.ylabel("Confirmed Cases")
plt.xlabel("Days")
plt.legend()


# **Simple Neural Network forecasting**

# In[ ]:


tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(150, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.Huber(), optimizer="adam")
model.fit(dataset,epochs=100)


# In[ ]:


forecast=[]
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
  

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]
print(forecast)

plt.figure(figsize=(10, 6))
plt.title("New York SNN Forecasting")
plt.ylabel("Confirmed Cases")
plt.xlabel("Days")

plt.plot( time_valid, x_valid,label="Real")
plt.plot(time_valid, results,label="Forecasted")
plt.legend()


# **Bidirectional LSTM forecasting**

# In[ ]:


tf.keras.backend.clear_session()
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True,activation="relu")),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,activation="relu")),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])


model.compile(loss=tf.keras.losses.Huber(), optimizer="adam",metrics=["mae"])
history = model.fit(dataset,epochs=1000)


# In[ ]:


forecast=[]
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]
print(forecast)

plt.figure(figsize=(10, 6))

plt.plot( time_valid, x_valid,label="Real")
plt.plot(time_valid, results,label="Forecasted")
plt.title("Bidirectionl LSTM Forecasting")
plt.ylabel("Confirmed Cases")
plt.xlabel("Days")


# **Bidirectional LSTM seems to give most accurate results on validation set. This neural network will be trained with full train set and then will be used to forecast on test set.**

# In[ ]:


df1=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")


# In[ ]:


df1.info()


# In[ ]:


df1=df1[df1["Province_State"]=="New York"]


# In[ ]:


df1.info()


# In[ ]:


df2=pd.DataFrame()


# In[ ]:


df2["days_test"]=[x for x in range(1,85)]


# In[ ]:


#df3=df.reset_index()
#df2=df2.reset_index()
df4 = [df3, df2]
df_test = pd.concat(df4, axis=1)


# In[ ]:


df_test.info()


# In[ ]:


df_test.tail(20)


# In[ ]:


x=df_test.iloc[:,2]
time=df_test.iloc[:,8]
y=df_test.iloc[:,5]
time=time.to_numpy(dtype="float32")
series=y.to_numpy(dtype="float32")


# In[ ]:


time=np.array(time)
series=np.array(series)
split_time = 72
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


# In[ ]:


window_size = 2
batch_size = 3
shuffle_buffer_size = 74


# In[ ]:


print(series)


# In[ ]:


forecast=[]
for time in range(len(series) - window_size):
    print(time)
    z=model.predict(series[time:time + window_size][np.newaxis])
    print(z)
    if time >= 72:
        series[time+window_size]=z
        print(series)
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
 
  


forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(20, 10))

plt.plot( time_valid, x_valid,label="Real")
plt.plot(time_valid, results,label="Forecasted")
plt.title("Bidirectionl LSTM Forecasting")
plt.ylabel("Confirmed Cases")
plt.xlabel("Days")


# In[ ]:


print(forecast)

