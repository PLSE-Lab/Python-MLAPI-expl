#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)


# In[ ]:


#from google.colab import files
#files.upload()


# In[ ]:


# last 5 year data of Apple stock market
data = pd.read_csv('../input/AAPL.csv')


# In[ ]:


data.head()


# In[ ]:


plt.figure(figsize=(15,10))  
plt.plot(data['Open'], color='blue', label='Apple Open Stock Price')  
plt.title('Apple Stock Market Open Price vs Time')  
plt.xlabel('Date')  
plt.ylabel('Apple Stock Price')  
plt.legend()  
plt.show()  


# In[ ]:


data['Date'] = pd.to_datetime(data['Date']) 
data.head()


# In[ ]:


X = np.array(data['Open'])
X = X.reshape(X.shape[0],1)
X.shape


# In[ ]:


from sklearn.preprocessing import MinMaxScaler  
scaler = MinMaxScaler(feature_range = (0, 1))
X = scaler.fit_transform(X) 


# In[ ]:


tp=20
train = X[:1150]
test = X[1150-tp:]


# In[ ]:


print(train.shape,'\n',test.shape)


# In[ ]:


#Create X_train using 30 timesteps for each sample
X_train = []
y_train = []
for i in range(tp, train.shape[0]):
    X_train.append(train[i-tp:i, 0])
    y_train.append(train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[ ]:


print(X_train.shape,'\n',y_train.shape)


# In[ ]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  
X_train.shape


# In[ ]:


from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense  
from tensorflow.keras.layers import LSTM  
from tensorflow.keras.layers import Dropout  


# In[ ]:


from tensorflow.keras import backend


# In[ ]:


model = Sequential()  


# In[ ]:


# Training LSTM model

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
lstm_model = Sequential()
lstm_model.add(LSTM(12, input_shape=(X_train.shape[1], 1), activation='relu',kernel_initializer='lecun_uniform',return_sequences=True))
lstm_model.add(LSTM(12, activation='relu',kernel_initializer='lecun_uniform'))
lstm_model.add(Dense(1))


# In[ ]:


lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
 


# In[ ]:


lstm_model.fit(X_train, y_train, epochs = 50, batch_size = 4)


# In[ ]:


# Create X_test using 30 timesteps for each sample
X_test = []
y_test = []

for i in range(tp, test.shape[0]):
    X_test.append(test[i-tp:i, 0])
    y_test.append(test[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
print(X_test.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_train.shape)


# In[ ]:


# plot predictions vs real turnover on training set
plt.figure(figsize=(15,10))
predicted = lstm_model.predict(X_train)
predicted = scaler.inverse_transform(predicted)
plt.plot(scaler.inverse_transform(train[-X_train.shape[0]-1:]), color = 'red', label = 'Open Price')
plt.plot(predicted, color = 'green', label = 'Predicted Open Price')
plt.title('Apple Stock Market Open Price vs Time')
plt.xlabel('Time')
plt.ylabel('Open Price')
plt.legend()
plt.show()


# In[ ]:


X_test.shape


# In[ ]:


# plotting predictions vs true turnover for the test set
plt.figure(figsize=(15,10))
predicted = lstm_model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
plt.plot(scaler.inverse_transform(test[-X_test.shape[0]-1:]), color = 'red', label = 'Open Price')
plt.plot(predicted, color = 'green', label = 'Predicted Open Price')
plt.title('Apple Stock Market Open Price vs Time')
plt.xlabel('Time')
plt.ylabel('Open Price')
plt.legend()
plt.show()


# In[ ]:




