#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
print(os.listdir("../input"))
import keras


# In[ ]:


# Read data from csv file, consider column 'date' as index and parse it into date object
dataset = pd.read_csv('../input/historical_stock_prices.csv', index_col='date', parse_dates=['date'])


# In[ ]:


# Display some sort of sample of dataset
dataset.head()


# In[ ]:


# Selecting training_set from begin of 2015 till end of 2017, we can do this because our index in date object
training_data = dataset['2015':'2017'].adj_close


# In[ ]:


training_data.shape


# In[ ]:


# Verify training_data corresponding to our selection or not
training_data.head(-1)


# In[ ]:


# transform method required reshape our training_set
training_set = training_data.values.reshape(-1, 1)


# In[ ]:


# It is kind of data preprocessing in order to obtain more accurancy of our model
# Once I trained our model without transform training set I got higher loss
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[ ]:


training_set_scaled.shape


# In[ ]:


# we create a data structure with 60 timesteps and 1 output
# We consider 60 previous elements to produce 1 output
# You can set beside from 60 timesteps but my case I prefer 60 timesteps
X_train = []
y_train = []
timesteps = 60
for i in range(timesteps,len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i,0])
# y_train[0] is an output of X_train[0], X_train[0] contains 60 previous elements of y_train[0]


# In[ ]:


# Frame our X_train and y_train into numpy array
X_train, y_train = np.array(X_train), np.array(y_train)


# In[ ]:


# Reshaping X_train to match with our GRU network input
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))


# In[ ]:


X_train.shape


# In[ ]:


# Callback function `earlystopping` to trigger while training it monitor 'loss'
# start from first epoch it record min value of loss and if model can not find 
# lower loss value in next 3 epochs model will stop training.
es = keras.callbacks.EarlyStopping(monitor='loss',patience=3, mode='min')


# In[ ]:


# The GRU architecture
# We are using Sequential model of Keras
# Our hidden layer using activation = 'tanh' and having 50 neurons each layer
# The output layer of fully connected layer is having only 1 neurons because we want only one output value

regressorGRU = Sequential()
# First GRU layer with Dropout regularisation
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Fourth GRU layer
regressorGRU.add(GRU(units=50, activation='tanh'))
regressorGRU.add(Dropout(0.2))
# The output layer
regressorGRU.add(Dense(units=1))
# Compiling the RNN
regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
# Fitting to the training set


# In[ ]:


# Training model with earlystopping callbacks function
# It's processing 6000 sample a step
regressorGRU.fit(X_train,y_train,epochs=20,batch_size=6000, callbacks=[es])


# # Calculate MSE #

# In[ ]:


def return_mse(test,predicted):
    mse = mean_squared_error(test, predicted)
    print("The mean squared error is {}.".format(mse))


# ## Calculate RMSE on test set ##

# In[ ]:


# Selecting test set
# Consider index start from 2018 date object and select only adj_close column
test_set = dataset['2018':].adj_close.values


# In[ ]:


test_set.shape


# In[ ]:


total_dateset = dataset.adj_close
inputs = total_dateset[len(total_dateset.values)-len(test_set)-60:].values


# In[ ]:


inputs.shape


# In[ ]:


# Reshape inputs and scale it because our training set also scaled
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)


# In[ ]:


# Frame X_test and y_test structure same as trining
X_test = []
y_test = []
for i in range(timesteps,len(inputs)):
    X_test.append(inputs[i-timesteps:i,0])
    y_test.append(inputs[i,0])
X_test = np.array(X_test)
Y_test = np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


# In[ ]:


# Selecting some from test set
test = X_test[:10000]
y = Y_test[:10000]


# In[ ]:


# Predicting
predicted_stock_price = regressorGRU.predict(test)


# In[ ]:


predicted_stock_price


# In[ ]:


y


# In[ ]:


# Evaluating our model
return_mse(y,predicted_stock_price)

