#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Dependencies

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


# Get Data
data = pd.read_csv('../input/google-stock-market-data/GOOG.csv', date_parser = True)
data.head(10)


# In[ ]:


data_training = data[data['Date']<'2019-01-01'].copy()
data_test = data[data['Date']>='2019-01-01'].copy()


# In[ ]:


data_training = data_training.drop(['Date', 'Adj Close'], axis = 1)


# In[ ]:


scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)
data_training[0:5]


# In[ ]:


X_train = []
y_train = []


# In[ ]:


for i in range(60, data_training.shape[0]):
    X_train.append(data_training[i-60:i])
    y_train.append(data_training[i, 0])


# In[ ]:


X_train, y_train = np.array(X_train), np.array(y_train)


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# **Building the Neural Network**

# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


# In[ ]:


model = Sequential()

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(units = 1))


# In[ ]:


model.compile(optimizer='adam', loss = 'mean_squared_error')


# In[ ]:


model.fit(X_train, y_train, epochs=5, batch_size=32)


# In[ ]:


data_test.head()


# In[ ]:


data_training = data[data['Date']<'2019-01-01'].copy()

past_60_days = data_training.tail(60)

df = past_60_days.append(data_test, ignore_index = True)
df = df.drop(['Date', 'Adj Close'], axis = 1)
df.head()


# In[ ]:


inputs = scaler.transform(df)
inputs


# In[ ]:



X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])


# In[ ]:



X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


scaler.scale_


# In[ ]:


scale = 1/8.18605127e-04
scale


# In[ ]:


y_pred = y_pred*scale
y_test = y_test*scale


# In[ ]:


# Visualising the results
plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

