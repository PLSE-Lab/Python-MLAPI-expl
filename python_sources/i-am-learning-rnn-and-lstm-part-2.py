#!/usr/bin/env python
# coding: utf-8

# # Content
# * Introduction
# * Import Library
# * Load Data
# * Lstm
# * Rnn
# * Conclusion

# ### Introduction
# * I try my first Recurrent Neural Network.
# * This work is a step towards my goal of becoming an artificial intelligence developer.
# * With the data before, I've also trained and tested with the models I created with simple rnn.
# * I wish good readings. See you below to evaluate the results.

# In[ ]:


# import library
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SimpleRNN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout, Flatten
import os
print(os.listdir("../input"))


# In[ ]:


# import data
data = pd.read_csv('../input/international-airline-passengers.csv',skipfooter=5, engine='python')
data.head()


# In[ ]:


# data plotting
dataset = data.iloc[:,1].values
plt.plot(dataset)
plt.xlabel("time")
plt.ylabel("Number of Passenger")
plt.title("international airline passenger")
plt.show()


# In[ ]:


# data shape (142,), reshape (142,1) 
dataset = dataset.reshape(-1,1,)
dataset = dataset.astype("float32")
dataset.shape


# * Reduce the values between o and 1.
# * In this way, we normalize the data and provide ease of operation.

# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# * We create training and test data.

# In[ ]:


train_size = int(len(dataset) * 0.50)
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
print("train size: {}, test size: {} ".format(len(train), len(test)))


# * Our training and test data are divided into x and y.

# In[ ]:


time_stemp = 10
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = np.array(dataX)
trainY = np.array(dataY)


# In[ ]:


dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = np.array(dataX)
testY = np.array(dataY)


# * We reshape our data for the Keras library.

# In[ ]:


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# * We build our LSTM model.

# In[ ]:


model_lstm = Sequential()
model_lstm.add(LSTM(100,return_sequences=True,input_shape=(1, time_stemp) ))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(80, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(50))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer = 'adam')
model_lstm.fit(trainX, trainY, epochs = 250, batch_size = 32)


# * We are testing the model with the test data.
# * We return the data back to its actual values.
# * We print the mean squared error rate of our estimates with training and test data.

# In[ ]:


trainPredict = model_lstm.predict(trainX)
testPredict = model_lstm.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# * We plot the results.

# In[ ]:


trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[ ]:


data = pd.read_csv('../input/international-airline-passengers.csv',skipfooter=5, engine='python')
data.head()


# In[ ]:


dataset = data.iloc[:,1].values
plt.plot(dataset)
plt.xlabel("time")
plt.ylabel("Number of Passenger")
plt.title("international airline passenger")
plt.show()


# In[ ]:


dataset = dataset.reshape(-1,1)
dataset = dataset.astype("float32")
print(dataset.shape)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = scaler.fit_transform(dataset)
#print(dataset_scaled)


# In[ ]:


plt.plot(dataset_scaled)
plt.show()


# In[ ]:


train_size = int(len(dataset_scaled) * 0.60)
test_size = len(dataset_scaled) - train_size
train = dataset_scaled[0:train_size,:]
test = dataset_scaled[train_size:len(dataset_scaled),:]
print("train size: {}, test size: {} ".format(len(train), len(test)))


# In[ ]:


time_stemp = 10
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = np.array(dataX)
trainY = np.array(dataY)  
trainX.shape


# In[ ]:


dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = np.array(dataX)
testY = np.array(dataY) 
testX.shape


# In[ ]:


trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
print('shape of trainX: ', trainX.shape)
print('shape of testX: ', testX.shape)


# * Rnn model create

# In[ ]:


model_rnn = Sequential()

model_rnn.add(SimpleRNN(units = 100, activation = 'relu', return_sequences = True, input_shape = (trainX.shape[1], 1)))
model_rnn.add(Dropout(0.25))

model_rnn.add(SimpleRNN(units = 70, activation = 'relu', return_sequences = True))
model_rnn.add(Dropout(0.25))

model_rnn.add(SimpleRNN(units = 60, activation = 'relu', return_sequences = True))
model_rnn.add(Dropout(0.25))

model_rnn.add(SimpleRNN(units = 30, activation = 'relu', return_sequences = True))
model_rnn.add(Dropout(0.25))

model_rnn.add(SimpleRNN(units = 20))
model_rnn.add(Dropout(0.25))
model_rnn.add(Dense(units = 1))

model_rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
model_rnn.fit(trainX, trainY, epochs = 250, batch_size = 32)


# In[ ]:


trainPredict = model_rnn.predict(trainX)
testPredict = model_rnn.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[ ]:


trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict

plt.plot(scaler.inverse_transform(dataset_scaled), color='b')
plt.plot(trainPredictPlot, color='r')
plt.plot(testPredictPlot, color='g')
plt.show()


# ### Conclusion
# * My first rnn try.
# * I got better results than simple rnn with lstm.
# * Your comments are very important to me.
# * Thank you for reading my notebook and for comments.
