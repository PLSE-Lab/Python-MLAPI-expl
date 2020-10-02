#!/usr/bin/env python
# coding: utf-8

# # Stock Market Prediction (Google)

# **Stock market prediction** is the act of trying to determine the future value of a company stock or other financial instrument traded on an exchange. The successful prediction of a stock's future price could yield significant profit. The efficient-market hypothesis suggests that stock prices reflect all currently available information and any price changes that are not based on newly revealed information thus are inherently unpredictable. Others disagree and those with this viewpoint possess myriad methods and technologies which purportedly allow them to gain future price information.

# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


train_dataset = pd.read_csv('../input/google_stock_train.csv')
test_dataset = pd.read_csv('../input/google_stock_test.csv')
train_data = train_dataset.iloc[:, 1:2].values


# In[ ]:


min_max = MinMaxScaler(feature_range = (0, 1))
train_data_scaled = min_max.fit_transform(train_data)


# In[ ]:


X_train = []
y_train = []

for i in range(60, len(train_data_scaled)):
  X_train.append(train_data_scaled[i-60:i, 0])
  y_train.append(train_data_scaled[i, 0])
  
X_train, y_train = np.array(X_train), np.array(y_train)


# In[ ]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences = True, input_shape = (X_train.shape[1],1)),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.LSTM(50, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.LSTM(50, return_sequences = True),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(1)
])


# In[ ]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[ ]:


model.fit(X_train, y_train, epochs = 100, batch_size = 32)


# In[ ]:


real_data = test_dataset.iloc[:, 1:2].values


# In[ ]:


total_data = pd.concat((train_dataset['Open'], test_dataset['Open']), axis=0)
inputs = total_data[len(total_data)-len(test_dataset)-60:].values
inputs = inputs.reshape(-1,1)
inputs = min_max.transform(inputs)


# In[ ]:


X_test = []
for i in range(60,123):
  X_test.append(inputs[i-60:i, 0])
  
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predict_data = model.predict(X_test)
predict_data = min_max.inverse_transform(predict_data)


# In[ ]:


plt.plot(real_data, color="red", label="Real Google Stock Price")
plt.plot(predict_data, color="blue", label="Predicted Google Stock Price")
plt.title("Google Stock Prize Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Prize")
plt.legend()
plt.show()

