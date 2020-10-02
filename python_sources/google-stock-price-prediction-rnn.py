#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset_train = pd.read_csv("../input/trainset.csv")


# In[ ]:


dataset_train


# In[ ]:


trainset = dataset_train.iloc[:,1:2].values


# In[ ]:


trainset


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_scaled = sc.fit_transform(trainset)


# In[ ]:


training_scaled


# In[ ]:


x_train = []
y_train = []


# In[ ]:


for i in range(60,1259):
    x_train.append(training_scaled[i-60:i, 0])
    y_train.append(training_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)


# In[ ]:


x_train.shape


# In[ ]:


x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[ ]:


regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))


# In[ ]:


regressor.add(Dropout(0.2))


# In[ ]:


regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))


# In[ ]:


regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))


# In[ ]:


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# In[ ]:


regressor.add(Dense(units = 1))


# In[ ]:


regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')


# In[ ]:


regressor.fit(x_train,y_train,epochs = 100, batch_size = 32)


# In[ ]:


dataset_test =pd.read_csv("../input/testset.csv")


# In[ ]:


real_stock_price = dataset_test.iloc[:,1:2].values


# In[ ]:


dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
dataset_total


# In[ ]:


inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs


# In[ ]:


inputs = inputs.reshape(-1,1)


# In[ ]:


inputs


# In[ ]:


inputs = sc.transform(inputs)
inputs.shape


# In[ ]:


x_test = []
for i in range(60,185):
    x_test.append(inputs[i-60:i,0])


# In[ ]:


x_test = np.array(x_test)
x_test.shape


# In[ ]:


x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
x_test.shape


# In[ ]:


predicted_price = regressor.predict(x_test)


# In[ ]:


predicted_price = sc.inverse_transform(predicted_price)
predicted_price


# In[ ]:


plt.plot(real_stock_price,color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

