#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#changing the Directory where data is located
import os
os.chdir('/kaggle/input/air-passengers')
os.getcwd()


# importing required libraries.

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[ ]:


data = pd.read_csv('AirPassengers.csv')
data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# Changing month to datetime format

# In[ ]:


data['Month'] = pd.to_datetime(data['Month'])
data.info()


# In[ ]:


data.set_index('Month', inplace=True) #set date as index


# In[ ]:


data.head()


# Plotting the Time Series Graph.

# In[ ]:


plt.xlabel("Month")
plt.ylabel("Passengers")
plt.title("Passengers Travelled")
plt.plot(data['#Passengers'],)


# Scaling the Data

# In[ ]:



#data_Scaling
data['#Passengers']

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data=scaler.fit_transform(data)


# In[ ]:


data


# In[ ]:



#Creating train and test partition
train = int(len(data)*0.75)
test = len(data)-train

train

test

train_data,test_data=data[0:train,:],data[train:len(data),:1]


# In[ ]:


# converting an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


# In[ ]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 4
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

y_train


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:



# reshape input to be [sample, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0], 4, 1)
X_test = X_test.reshape(X_test.shape[0], 4, 1)

X_train.shape


# In[ ]:



### Create the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(4,1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=1,verbose=1)


# In[ ]:


#Model Prediction
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

test_predict


# In[ ]:


#Transforming data back to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

test_predict


# In[ ]:


## Calculate RMSE performance metrics
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[ ]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[ ]:


### Plotting 
# shift train predictions for plotting
look_back=4
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(data)-1, :] = test_predict
# plot baseline and predictions
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.title("Passengers Travelled")
plt.plot(scaler.inverse_transform(data)) #original data
plt.plot(testPredictPlot) #test prediction
plt.show()

