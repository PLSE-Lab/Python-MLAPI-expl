#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION**
# 
# In th's project, we'll try to understand and exercise the RNN (Recurrent Neural Network). I decided to use 'International Airline Passengers' dataset for this workout. You'll see:
# * Data Loading and Reading
# * Data Preprocessing (Scaling, Train-Test Datas Split, Reshaping)
# * RNN with Keras
# * Predictions and Visualizations
# * Conclusion
# 
# in this project.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Firstly, read data from csv file.
dataset = pd.read_csv('../input/international-airline-passengers.csv',skipfooter=5)


# In[ ]:


dataset.info()


# In[ ]:


dataset.head()


# In[ ]:


dataset.tail(10)


# In[ ]:


dataset.describe()


# In[ ]:


# We only use Number of Passengers in this project. Therefore, we create a new data named as 'data' and
# assign to just passenger number to this new smaller data.
data = dataset.iloc[:,1].values


# In[ ]:


# Let's take a look our new data.
plt.plot(data)
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.title("International Airline Passengers")
plt.show()


# In[ ]:


# Let's look at the shape of data.
data.shape


# In[ ]:


# As you can see; shape of data is (142,). We should reshape it.
data =data.reshape(-1,1)
data.astype("float32")
data.shape


# In[ ]:


# After reshaping, we should scale all of datas between 0 and 1.
from sklearn.preprocessing import MinMaxScaler #import scling library
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)


# In[ ]:


# Let's check our data!
data_scaled
# As you can see, we scaled our values!


# **and now time to train-test datas split!**

# In[ ]:


train_data_size = int(len(data_scaled)*0.50)
test_data_size = len(data_scaled) - train_data_size
print("Train data size is {}".format(train_data_size))
print("Test data size is {}".format(test_data_size))


# In[ ]:


train = data_scaled[0:train_data_size,:]
test = data_scaled[train_data_size:len(data_scaled),:]
# Let's check number of train and test datas again
print("Train data size is {}".format(len(train)))
print("Test data size is {}".format(len(test)))


# In[ ]:


x_train = []
y_train = []
time_steps=10
for i in range(len(train)-time_steps-1):
    a = train[i:(i+time_steps),0]
    x_train.append(a)
    y_train.append(train[i + time_steps,0])
trainX = np.array(x_train)
trainY = np.array(y_train)


# In[ ]:


trainX.shape


# In[ ]:


x_test = []
y_test = []
for i in range(len(test)-time_steps-1):
    a = test[i:(i+time_steps),0]
    x_test.append(a)
    y_test.append(test[i + time_steps,0])
testX = np.array(x_test)
testY = np.array(y_test)


# In[ ]:


testX.shape


# In[ ]:


# Let's reshape trainX and testX
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
testX = np.reshape(testX, (testX.shape[0],testX.shape[1],1))
# Print and check shapes
print("Shape of trainX is {}".format(trainX.shape))
print("Shape of testX is {}".format(testX.shape))


# **Time to RNN with Keras!!**

# In[ ]:


# Firstly, define libraries
from keras.layers import Dense, SimpleRNN, Dropout
from keras.metrics import mean_squared_error
from keras.models import Sequential


# In[ ]:


# Initializing RNN
model = Sequential()
# Add the first layer and Dropout regularization
model.add(SimpleRNN(units=100,activation='tanh',return_sequences=True, 
                    input_shape=(trainX.shape[1],1)))
model.add(Dropout(0.20))
# Second layer and Dropout regularization
model.add(SimpleRNN(units = 100, activation='tanh',return_sequences=True))
model.add(Dropout(0.20))
# Third layer and Dropout regularization
model.add(SimpleRNN(units = 70, activation='tanh', return_sequences= True))
model.add(Dropout(0.20))
# Fourth layer and Dropout regularization
model.add(SimpleRNN(units = 50))
model.add(Dropout(0.20))
# Add final or output layer
model.add(Dense(units=1))

# Compile our RNN model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the training set
model.fit(trainX, trainY, epochs = 200, batch_size=32)
# Remember; epochs, batch_size etc. are just some of hyper parameters. 
# You can change these parameters whatever you want


# **And the final mission: Predictions and Visualization!**

# In[ ]:


trainPrediction = model.predict(trainX)
testPrediction = model.predict(testX)

# Remember, we scaled datas between 0 and 1 but now we're at the end of the project.
# So we should inverse transform datas.

trainPrediction = scaler.inverse_transform(trainPrediction)
trainY = scaler.inverse_transform([trainY])
testPrediction = scaler.inverse_transform(testPrediction)
testY = scaler.inverse_transform([testY])


# Let's look at our RMSE (Root Mean Squared Error)

# In[ ]:


# There is some problem in there but I didn't know what is it.
# I googled it and helps with DATAI Team found the problem :)
# Convert tensor to numpy. Otherwise we could not sqrt values.
import tensorflow as tf
sess = tf.Session()
with sess.as_default():
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPrediction[:,0]).eval())
    testScore = math.sqrt(mean_squared_error(testY[0], testPrediction[:,0]).eval())
print("Train Score is %.2lf RMSE"%(trainScore))
print("Test Score is %.2lf RMSE"%(testScore))


# In[ ]:


trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_steps:len(trainPrediction)+time_steps, :] = trainPrediction

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPrediction)+(time_steps*2)+1:len(dataset)-1, :] = testPrediction

plt.plot(scaler.inverse_transform(data_scaled),label = 'True Values', color='blue')
plt.plot(trainPredictPlot,label='Train Prediction', color='red')
plt.plot(testPredictPlot,label = 'Test Prediction', color='green')
plt.xlabel("Time")
plt.ylabel("Number of Passengers")
plt.title("International Airline Passengers")
plt.legend()
plt.show()


# **CONCLUSION**
# 
# We tried to workout on RNN with Keras library. Our predictions are not perfect but that seems enough for now :) I learned a lot of things from this project myself and I hope you did it too. Your examinations and comments are so important to me. Please, check it out and feel relax to comment me. 
