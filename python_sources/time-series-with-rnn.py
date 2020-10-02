#!/usr/bin/env python
# coding: utf-8

# Thankful to The TensorFlow Authors.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Time series forecasting

# This tutorial is an introduction to time series forecasting using Recurrent Neural Networks (RNNs). This is covered in two parts: first, you will forecast a univariate time series, then you will forecast a multivariate time series.

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


# In[ ]:


ops=pd.read_csv("/kaggle/input/opsd_germany_daily.csv")


# In[ ]:


uni_data = ops['Consumption']
uni_data.index = ops['Date']
uni_data.head()


# In[ ]:


def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)


# In[ ]:


uni_data.plot(subplots=True)


# In[ ]:


TRAIN_SPLIT = 960


# In[ ]:


tf.random.set_seed(13)


# In[ ]:


uni_data = uni_data.values


# In[ ]:


uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()


# Let's standardize the data.

# In[ ]:


uni_data = (uni_data-uni_train_mean)/uni_train_std


# Let's now create the data for the univariate model. For part 1, the model will be given the last 20 recorded temperature observations, and needs to learn to predict the temperature at the next time step. 

# In[ ]:


univariate_past_history = 8
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)


# This is what the `univariate_data` function returns.

# In[ ]:


x_train_uni.shape


# In[ ]:


print ('Single window of past history')
print (x_train_uni[0])
print ('\n Target temperature to predict')
print (y_train_uni[0])


# Now that the data has been created, let's take a look at a single example. The information given to the network is given in blue, and it must predict the value at the red cross.

# ### Recurrent neural network
# 
# A Recurrent Neural Network (RNN) is a type of neural network well-suited to time series data. RNNs process a time series step-by-step, maintaining an internal state summarizing the information they've seen so far. For more details, read the [RNN tutorial](https://www.tensorflow.org/tutorials/sequences/recurrent). In this tutorial, you will use a specialized RNN layer called Long Short Term Memory ([LSTM](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/LSTM))
# 
# Let's now use `tf.data` to shuffle, batch, and cache the dataset.

# You will see the LSTM requires the input shape of the data it is being given.

# In[ ]:


simple_RNN_model = tf.keras.models.Sequential([
    tf.keras.layers.SimpleRNN(20,return_sequences=True, input_shape=x_train_uni.shape[-2:]),
    keras.layers.SimpleRNN(20),
    tf.keras.layers.Dense(1)
])

simple_RNN_model.compile(optimizer='adam', loss='mae')


# Let's make a sample prediction, to check the output of the model. 

# Let's train the model now. Due to the large size of the dataset, in the interest of saving time, each epoch will only run for 200 steps, instead of the complete training data as normally done.

# In[ ]:


history = simple_RNN_model.fit(x_train_uni, y_train_uni, epochs=100,
                     batch_size=30,
                      validation_data=(x_val_uni, y_val_uni))


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0.15, .5) # set the vertical range to [0-1]
plt.show()


# In[ ]:


def prediction_plot(testY, test_predict):
     len_prediction=[x for x in range(len(testY))]
     plt.figure(figsize=(8,4))
     plt.plot(len_prediction, testY, marker='.', label="actual")
     plt.plot(len_prediction, test_predict, 'r', label="prediction")
     plt.tight_layout()
     sns.despine(top=True)
     plt.subplots_adjust(left=0.07)
     plt.ylabel('Power Consumption', size=15)
     plt.xlabel('Time step', size=15)
     plt.legend(fontsize=15)
     plt.show();


# In[ ]:


y_pred = simple_RNN_model.predict(x_val_uni)
prediction_plot(y_val_uni,y_pred)

