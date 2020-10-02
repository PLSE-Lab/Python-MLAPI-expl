#!/usr/bin/env python
# coding: utf-8

# # Global Confirmed Case Prediction
# by Zixuan Jin(z77jin), Ran Zang(rzang)
# 
A Recurrent Neural Network (RNN) is a type of neural network well-suited to time series data. RNNs process a time series step-by-step, maintaining an internal state summarizing the information they've seen so far. And in this assignment we use the simple LSTM model to predict the global confirmed case. 
# **1. Getting Data**
# 
# Fisrt, we import the data and extract the daily global confirmed cases from the data.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
# set the random_seed to reproduce our work
tf.random.set_seed(13)
# load data
data = pd.read_csv("../input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv")


# We remove all the provinces number and add up all the confirmed cases and get the global data as follows.

# In[ ]:


data = data.fillna(0)
data = data[data['Province/State']==0]
global_data = data.sum()[4:].reset_index()
global_data.columns=['date','number']
global_data['date'] = global_data['date'].map(lambda x:str(x)[:-3])
all_data = global_data['number']
all_data.index = global_data['date']
global_data


# In[ ]:


all_data.plot(color='black')
plt.title("Global comfirmed number")
plt.show()


# We split the data into train and test set, where train set is the global confirmed number from 1/22 to 3/31 and test set is from 4/1 to 4/17.

# **2. Normalization**
# 
# We do a z-score normalization to the data and define a recover function to get the original data.

# In[ ]:


TRAIN_SPLIT = 70
all_data = all_data.values
train_mean = all_data[:TRAIN_SPLIT].mean()
train_std = all_data[:TRAIN_SPLIT].std()


# In[ ]:


def normalize(target):
    return (target-train_mean)/train_std

def recover(target):
    return target*train_std + train_mean


# In[ ]:


all_data = normalize(all_data)


# Define a function to get the corresponding data and label

# In[ ]:


def split_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        indices = range(i-history_size,i)
        data.append(np.reshape(dataset[indices],(history_size,1)))
        labels.append(dataset[i+target_size])
    return np.array(data,dtype=np.float),np.array(labels,dtype=np.float)


# In[ ]:


x_train,y_train = split_data(all_data,0,TRAIN_SPLIT,7,0)
x_val,y_val = split_data(all_data, TRAIN_SPLIT, None, 7,0)


# In[ ]:


BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


# **3.Construct LSTM model**
# 
# We select parameter of units=50,activation='tanh', activation='tanh',optimizer='adam' on the LSTM model.

# In[ ]:


simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, input_shape=x_train.shape[-2:],activation='relu'),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')


# **4. Fit the model**

# In[ ]:


EPOCHS = 5
history = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=200,
                      validation_data=val_univariate, validation_steps=50, verbose=1)


# In[ ]:


epochs = list(range(EPOCHS))
plt.plot(epochs,history.history['loss'],label='loss',color='black',linestyle='--',linewidth=1)
plt.plot(epochs,history.history['val_loss'],label='val_loss',color='black',linestyle='-',linewidth=1)
plt.title("Loss during fitting")
plt.legend()
plt.show()


# **5.Make predictions**
# 
# Use the 7 days before the date to make prediction and recursively make the whole prediction. Plot the predictions and the actual data as follows.

# In[ ]:


def makeprediction(sevendays):
    sample = np.array(sevendays, dtype=np.float)
    sample = np.reshape(sample,(1,7,1))
    return int(recover(simple_lstm_model.predict(sample))[0][0])


# In[ ]:


predictions = []
for i in range(50,79):
    sample = all_data[i:(i+7)]
    predictions.append(makeprediction(sample))


# In[ ]:


predictions = pd.DataFrame(data=predictions,columns=['number'],index=range(58,87))
plt.plot(global_data.index,global_data['number'],label='actual data',color='black',linestyle='-',linewidth=1)
plt.scatter(predictions.index,predictions['number'],label='predict data',color='black',marker='.',linewidth=1)
plt.title("Global confirmed number after 1/22")
plt.legend()
plt.show()


# It can be oberved from the plot that the predicted number are growing rapidly using the the simple LSTM model.
