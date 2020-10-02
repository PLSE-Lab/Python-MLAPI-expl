#!/usr/bin/env python
# coding: utf-8

# # AAPL Stock Analysis

# In[76]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


import os
print(os.listdir("../input/nyse"))


# In[78]:


# read csv file
# dataframe = pd.read_csv('../input/tsla-stock/TSLA.csv')
dataframe = pd.read_csv('../input/nyse/prices-split-adjusted.csv')
dataframe = dataframe[dataframe.symbol == 'AAPL']
dataframe = dataframe[['date','open','close']]
dataframe.columns = ['Time', 'First', 'Last']
dataframe['Time']=pd.to_datetime(dataframe['Time'],infer_datetime_format=True)
dataframe.head()


# ### Take Average value of opening and closing stock for smoothness  

# In[57]:


dataframe['Avg'] = (0.5*(dataframe['First'] + dataframe['Last']))
plot_df = dataframe[['Time', 'Avg']]
print(plot_df.set_index(plot_df.Time)['Avg'].plot())
dataset = dataframe.Avg.values
dataset


# ### Scaling the dataset 

# In[58]:


dataset = (0.5*(dataframe['First'] + dataframe['Last'])).values
print(dataset)
min_max_scaler = MinMaxScaler()
nparr = np.array([dataset])
dataset = min_max_scaler.fit_transform(np.reshape(nparr, (len(nparr[0]), 1)))
dataset = np.reshape(dataset, (1, len(dataset)))[0]
dataset


# ### Time delay function create time sequence of given delay

# In[59]:


def timeDelay(data, delay):
    X_data, y_data = [], []
    #naive version, vectorized version can be implemented,
    #but may run out of memory,
    for i in range(delay, len(data)):
        X_data.append(data[i - (delay):i].tolist())
    X_data = np.array(X_data)
    y_data = data[delay:]
    
    plt.plot(range(len(X_data.flatten())), X_data.flatten(), linestyle='solid', color='green', label='time delayed data')
    plt.plot(range(len(y_data)), y_data, linestyle='solid', color='blue', label='original data')
    
    return np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1)), np.reshape(y_data, (len(y_data),))


# ### Creating LSTM model

# In[60]:


def model():
    model = Sequential()
    model.add(LSTM(4,input_shape=(5, 1), dropout=0.2))
    model.add(Dense(1))
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print ("Compilation Time : ", time.time() - start)
    return model


# ### Define parameters

# In[62]:


ratio = 0.8 # Training test ratio
d = 5 # Delay
X, y = timeDelay(dataset, d)
n = int(ratio * len(X))


# In[63]:


X_train, y_train = X[:n], y[:n]
X_test, y_test = X[n:], y[n:]


# In[64]:


x_train_flattened = X_train.flatten()
x_test_flattened = X_test.flatten()
plt.figure(figsize=(25,10))
plt.plot(range(len(x_train_flattened)), x_train_flattened, linestyle='solid', color='green', label='train_data')
plt.plot(
    range(len(x_train_flattened), len(x_train_flattened) + len(x_test_flattened)), 
    x_test_flattened, linestyle='solid', color='blue', label='test_data')
plt.legend()


# In[65]:


model = model()


# In[66]:


model.summary()


# In[68]:


model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)


# In[69]:


baseline_testScore = min_max_scaler.inverse_transform(np.array([[mean_squared_error(y_test[10:], y_test[:-10])**.5]]))
trainScore = min_max_scaler.inverse_transform(np.array([[model.evaluate(X_train, y_train, verbose=0)**.5]]))
testScore = min_max_scaler.inverse_transform(np.array([[model.evaluate(X_test, y_test, verbose=0)**.5]]))

print('baseline test score = %.2f RMSE' % baseline_testScore)
print('train score = %.2f RMSE' % trainScore)
print('test score = %.2f RMSE' % testScore)


# In[70]:


# generate predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# In[71]:


y = np.append(y_train, y_test)
y_pred = np.append(y_train_pred, y_test_pred)


# In[ ]:


nn = 10000
plt.figure(figsize=(25,10))

plt.plot(dataframe['Time'][5:][-nn:], y[-nn:], linestyle='solid', color='green', label='train_data + test_data')
plt.plot(dataframe['Time'][5:][-nn:], y_pred[-nn:], linestyle='solid', color='blue', label='prediction')

plt.legend()


# # Possible Improvements
# * Check if more features can be extracted from the existing dataset (such as volume)
# * Try different Loss functions and activation functions to improve above forecast
# * Get stock data from actual APIs to get minute / second level data of selected ticker string
# * Get tweet data about the ticker's related company -> Get the sentiment of tweet -> Weight these sentiments based on the number of followers
# * (Can be inferred from tweets made by news companies) Fetch news articles for company -> Get sentiment of news article -> Weight based on the popularity of news article
