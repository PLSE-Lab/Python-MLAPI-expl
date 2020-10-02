#!/usr/bin/env python
# coding: utf-8

# # Bitcoin forecast with Keras
# An attempt to predict the price of Bitcoin with Keras.
# 
# But I got some Problems.
# 
# 
# # # Used Sources
# 
# * *1 This gave me apparently good results. [First steps and basis for forecasting](https://towardsdatascience.com/bitcoin-price-prediction-using-lstm-9eb0938c22bd)
# * *2 But then I read this and tried the improvements mentioned. [For corrections and optimizations](https://stackoverflow.com/questions/48760472/how-to-use-a-keras-rnn-model-to-forecast-for-future-dates-or-events)
# 
# Both of got me to the code and plot above.
# 
# 
# # # My questions now 
# * How to get better results? 
# * How to predict 1 to X days of the (real) future? How to get the "unknown"?
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import keras as ke
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

#from keras.models import Model
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM, CuDNNLSTM

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

print(tf.__version__)
print(ke.__version__)
print("ready")


# Reading CSV with Bitcoin-Close-Prices only.

# In[ ]:


df = pd.read_csv("../input/historical_data_btc_only_close.csv", delimiter=';')
df_norm = df

print(df_norm)


# Splitting into test and train.

# In[ ]:


prediction_days = 30

df_train= df[:len(df)-prediction_days]
df_test= df[len(df)-prediction_days:]


# Reshaping the sets like mentioned in "Used Sources" *2. 

# In[ ]:


training_set = df_train.values
training_set = min_max_scaler.fit_transform(training_set)

x_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
x_train = np.reshape(x_train, (1, len(x_train), 1))
y_train = np.reshape(y_train, (1, len(y_train), 1))

print(x_train.shape)
print(y_train.shape)
print(y_train)
print(x_train)


# In[ ]:


from tensorflow.python.compiler import tensorrt as trt

num_units = 4
activation_function = 'sigmoid'
optimizer = 'RMSProp'
loss_function = 'binary_crossentropy'
batch_size = 1 # Only 1 due to a problem where a size was requested by which the passed model can be divided. 
num_epochs = 50

# Initialize the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = num_units, 
                   #activation = activation_function, 
                   input_shape=(None, 1), 
                   stateful=False, 
                   return_sequences=True,
                   batch_input_shape=(1, None, 1)
#                    input_shape=(1, len(x_train), 1)
                  ))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = optimizer, loss = loss_function)

# Using the training set to train the model
import time
start = time.time()
regressor.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs)
end = time.time()
regressor.summary()
print(end - start)


# Shaping, scaling, starting prediction, printing the numbers. 

# In[ ]:


test_set = df_test.values
print(test_set.shape)
shape1 = test_set.shape[0] 
shape2 = test_set.shape[1]


inputs = np.reshape(test_set, (shape1, shape2))
print(inputs.shape)

inputs = min_max_scaler.transform(inputs)
print(inputs.shape)

inputs = np.reshape(inputs, (1, shape1, shape2))
print(inputs.shape)

predicted_price = regressor.predict(inputs)

print(predicted_price.shape)
# print((predicted_price))

predicted_price = np.reshape(predicted_price, (shape1, shape2))
print(predicted_price.shape)

predicted_price = min_max_scaler.inverse_transform(predicted_price)

print(predicted_price.size)
print(predicted_price)
print(test_set.size)

# regressor.predict_proba
for i in range(0, prediction_days):
    print(test_set[i][0],"-" , predicted_price[i][0],"\tDifference: " ,test_set[i][0]-predicted_price[i][0])


# In[ ]:


plt.figure(figsize=(25, 25), dpi=80, facecolor = 'w', edgecolor = 'k')

plt.plot(test_set[:, 0], color='red', label='Real BTC Price')
plt.plot(predicted_price[:, 0], color = 'blue', label = 'Predicted BTC Price')

plt.title('BTC Price Prediction', fontsize = 40)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize = 40)
plt.legend(loc = 'best')
plt.show()


# In[ ]:





# In[ ]:




