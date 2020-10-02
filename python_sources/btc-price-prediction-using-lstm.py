#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import warnings
warnings.filterwarnings(action='ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

# Library for Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


# Any results you write to the current directory are saved as output.
for i in os.listdir("../input"):
    print(i)

coinbase = pd.read_csv("../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11.csv")
cb_index = coinbase.index.ravel()

print(list(coinbase.columns), "\nSHAPE ::", coinbase.shape)
for i in coinbase.columns:
    print(i, "::", coinbase[i].dtype)

# Declaring number for random state for reproducibility
rstate=123
    
coinbase.describe()


# We choose Bitcoin price after 1.200.000 because this is the time where people starts to recongnize Bitcoin.  We map the `Close` and its changes in percentage.

# In[ ]:


a = coinbase["Close"][1200000:].fillna(method="backfill")


# In[ ]:


close_price = np.array(a).reshape(-1,1)
plt.figure(figsize=(14,6))
plt.title("Bitcoin Closing Price")
plt.grid()
plt.plot(close_price)

sc = MinMaxScaler()
close_priceSC = sc.fit_transform(close_price)
plt.figure(figsize=(14,6))
plt.title("Scaled")
plt.grid()
plt.plot(close_priceSC)


# Transforming time series data to trainable data

# In[ ]:


X = []
y = []
for i in range(60, len(close_priceSC)):
    X.append(close_priceSC[i-60:i, 0])
    y.append(close_priceSC[i,0])
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)


# Splitting train and test

# In[ ]:


X_train = X[:700000,:]
X_test = X[700000:,:]

y_train = y[:700000]
y_test = y[700000:]


# Plotting train and test data

# In[ ]:


plt.figure(figsize=(14,4))
plt.plot(range(700000),y_train)
plt.plot(range(700000, len(y)), y_test)
plt.legend(["Training", "Test"])
plt.grid()


# Preparing X for LSTM

# In[ ]:


X_train = X_train.reshape(-1,60,1)
X_test = X_test.reshape(-1,60,1)


# Train using LSTM 

# In[ ]:


get_ipython().run_cell_magic('time', '', "# The LSTM architecture\nregressor = Sequential()\n# First LSTM layer with Dropout regularisation\nregressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))\nregressor.add(Dropout(0.2))\n# Second LSTM layer\nregressor.add(LSTM(units=50, return_sequences=True))\nregressor.add(Dropout(0.2))\n# Third LSTM layer\nregressor.add(LSTM(units=50, return_sequences=True))\nregressor.add(Dropout(0.5))\n# Fourth LSTM layer\nregressor.add(LSTM(units=50))\nregressor.add(Dropout(0.5))\n# The output layer\nregressor.add(Dense(units=1))\n\n# Compiling the RNN\nregressor.compile(optimizer='adam', loss='mean_absolute_error')\n# Fitting to the training set\nregressor.fit(X_train, y_train, epochs=1, batch_size=500)")


# In[ ]:


y_pred = regressor.predict(X_test)
MSE = mean_absolute_error(y_test, y_pred)


# In[ ]:


plt.figure(figsize=(14,6))
plt.plot(sc.inverse_transform(y_test.reshape(-1,1)))
plt.plot(sc.inverse_transform(y_pred.reshape(-1,1)))
plt.title("Comparison with MAE {0:0.10f}".format(MSE))
plt.legend(["Y", "Prediction"])
plt.xlabel("Timeframe")
plt.ylabel("Price")


# This is clearly overfitting
