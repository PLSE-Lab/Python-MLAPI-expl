#!/usr/bin/env python
# coding: utf-8

# ## Importing neccessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


data = pd.read_csv('../input/air-passengers/AirPassengers.csv')
data.head()


# In[ ]:


data.rename(columns={'#Passengers':'passengers'},inplace=True)
data = data['passengers']
data=np.array(data).reshape(-1,1)


# In[ ]:


plt.plot(data)


# ## Feature Scaling

# In[ ]:


scaler= MinMaxScaler()
data=scaler.fit_transform(data)


# In[ ]:


train_size=100
test_size=44


# In[ ]:


train=data[0:train_size,:]
test=data[train_size:,:]


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


def get_data(data, look_back):
    dataX, dataY = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(data[i+look_back, 0])
    return np.array(dataX), np.array(dataY)


# In[ ]:


look_back = 1
X_train, y_train = get_data(train, look_back)


# In[ ]:


X_train.shape


# In[ ]:


X_test, y_test = get_data(test, look_back)


# In[ ]:


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[ ]:


X_train.shape


# In[ ]:


type(y_test)


# ## Building the LSTM

# In[ ]:


model = Sequential()
model.add(LSTM(5, input_shape = (1, look_back)))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train, y_train, epochs=25, batch_size=1)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


scaler.scale_


# In[ ]:


y_pred = scaler.inverse_transform(y_pred)


# In[ ]:


y_test = y_test.reshape(-1, 1)
y_test = scaler.inverse_transform(y_test)


# In[ ]:


# plot baseline and predictions
plt.figure(figsize=(14,5))
plt.plot(y_test, label = 'real number of passengers')
plt.plot(y_pred, label = 'predicted number of passengers')
plt.ylabel('# passengers')
plt.legend()
plt.show()


# In[ ]:




