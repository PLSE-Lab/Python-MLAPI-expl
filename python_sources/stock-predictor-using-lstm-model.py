#!/usr/bin/env python
# coding: utf-8

# ****1. IMPORTING THE LIBRARIES****

# In[ ]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time 
import math
df = pd.read_csv('../input/IBM.csv',delimiter=',')
df=df.set_index(['date'])
df.drop(df.columns[[5,6,7,9]],axis=1,inplace=True)
df.head(5)


# **2**.**DATA VISUALISATION** **AND ANALYSIS**

# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(20, 5))
plt.subplot(1,1,1)
plt.plot(df.open.values,color='blue',label='open')
plt.plot(df.close.values,color='green',label='close')
plt.plot(df.low.values,color='yellow',label='low')
plt.plot(df.high.values,color='red',label='high')
plt.plot(df.vwap.values,color='black',label='volume weighted average price')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
plt.show()


# In[ ]:


plt.figure(figsize=(10, 5))
plt.subplot(1,1,1)
plt.plot(df.volume.values,color='black',label='volume')
plt.title('stock Volume')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
plt.show()


# **3** **PRE PROCESSING**
#  
# Using **MinMaxScaler** to **Normalise **the data
# 
#  **Volume Weighted Average Price** is used for prediction.The rest of the attributes are only for *analysis and visualisation* purpose

# In[ ]:


df1=df
df.head()
df1.drop(df.columns[[0,1,2,3,4,6]],axis=1,inplace=True)
df1
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(df1)


# In[ ]:


train_size = int(len(df) * 0.70)
test_size = len(df) - train_size
train, test = df1[0:train_size,:], df1[train_size:len(df),:]
print(len(train),len(test))


# Splitting into *Test data and Train data* in the ratio of **70:30**.

# In[ ]:


def create(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# In[ ]:


look_back=1
trainX, trainY = create(train, look_back)
testX, testY = create(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
trainX.shape


# Converting dataframe into  a ***specific array structure (numpy)*** to feed the data into **LSTM**.

# **4 FITTING LSTM MODEL**

# In[ ]:


model = Sequential()
model.add(LSTM(5, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='RMSProp')
model.fit(trainX, trainY, epochs=100, batch_size=8, verbose=1)


# Training the **LSTM** model.
# 
# **Optimiser used**:**RMSProp**
# 
# **Loss Metrics**:**Root Mean Square Error**
# 
# About RMSProp brief:-
# The RMSprop optimizer is similar to the gradient descent algorithm with momentum. **The RMSprop optimizer restricts the oscillations in the vertical direction**. Therefore, we can increase our learning rate and our algorithm could take larger steps in the horizontal direction converging faster. The difference between RMSprop and gradient descent is on how the gradients are calculated.  The value of momentum is denoted by beta and is usually set to 0.9. 
# 

# **5 PREDICTION PHASE**

# In[ ]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
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


# **6** **PERFORMANCE ANALYSIS**
# 
# After inversion to the norminalised values calculating **RMSE** and displaying the error 

# In[ ]:



trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df1)-1, :] = testPredict

plt.figure(figsize=(20, 5))
plt.subplot(1,1,1)
plt.plot(scaler.inverse_transform(df1),color='blue',label='Actual Value')
plt.plot(trainPredictPlot,color='red',label='Train Prediction')
plt.plot(testPredictPlot,color='green',label='Test Prediction')
plt.title('Result')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')
plt.show()
predict=pd.DataFrame({'Predicted':testPredict[:,0]})
cc=df1[0:train_size,:]
predict.to_csv('output.csv')
predict.to_csv('given.csv')


# **7 CONCLUSION**

# Even though LSTM model fits good in the data prediction.There are many other factors stocks price depend upon such as **companies policies,countries economic policies,demonetisation,etc** .
