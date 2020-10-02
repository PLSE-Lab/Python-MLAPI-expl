#!/usr/bin/env python
# coding: utf-8

# Exploring time series forecasting with Keras. As usual, playing with the numbers just for fun.
# 
# I will start with a very basic approach to LSTM in Keras, under the assumption that sales by shop and item in November are correlated to sales in same shops and items in previous two months. This is a very simplified version of senkin13 (https://www.kaggle.com/senkin13) excellent work in (https://www.kaggle.com/senkin13/lstm-starter/code). All credit to senkin13.
# 
# I will look for other alternatives and corrections to improve the results. Comments and ideas will be welcome.

# Now, the usual stuff.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# I will use the convenient date_block_num for month number as my time series, and will convert from the time series into supervised learning.

# In[2]:


df_train = pd.read_csv(
    '../input/sales_train.csv',
    usecols = [1, 2, 3, 5]
).fillna(0)


# In[3]:


print (df_train.shape)
print (df_train.info())


# In[4]:


df_t = df_train.pivot_table(
    values='item_cnt_day',
    index=['shop_id', 'item_id'],
    columns='date_block_num',
    aggfunc='sum'
).fillna(0)


# In[7]:


print (df_t.head())
print (df_t.shape)


# Now, to create the supervised learning data, I create a df with the period-1 and period-2 monthly sales, as we are using the assumption that sales in November (block number 34) are related to previous two months sales. So the train df will include for each block number starting in 2, the sales from the two previous periods.
# 
# Months from 2 to 31 are used for training. For validation, I took two months, 32 and 33.

# In[8]:


X_train = pd.DataFrame()
y_train = pd.Series()
X_val = pd.DataFrame()
y_val = pd.Series()
for i in range(2, 32):
    print (i)
    print ('='*50)
    X = df_t.iloc[:, i-2:i]
    X.columns = ['P-2', 'P-1']
    X_train = X_train.append(X)
    y = pd.Series(df_t.iloc[:, i])
    y_train = y_train.append(y)
for j in range(32, 34):
    print (j)
    print ('='*50)
    X = df_t.iloc[:, i-2:i]
    X.columns = ['P-2', 'P-1']
    X_val = X_val.append(X)
    y = df_t.iloc[:, i]
    y_val = y_val.append(y)


# In[9]:


print (len(X_train), len(y_train))
print (len(X_val), len(y_val))


# The length of X_train and y_train and X_val and y_val are the same, which was expected but always good to see.

# LSTM model requires the input data (X) to be with a 3 dimension shape: samples, timesteps and features. I'm using two features, period-1 and period-2, so I reshape the data to make it fit with this.

# In[10]:


X_train2 = X_train.as_matrix()
print (X_train2)
X_train2 = X_train2.reshape(X_train2.shape[0], 1, X_train2.shape[1])
y_train2 = y_train.values
print (y_train2)


# In[11]:


X_val2 = X_val.as_matrix()
X_val2 = X_val2.reshape(X_val2.shape[0], 1, X_val2.shape[1])
y_val2 = y_val.values
print (X_val2)
print (y_val2)


# In[12]:


from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM


# So, the Keras model, with 64 neurons in the first layer, 10% drop to avoid overfitting, 32 neurons in a second layer, 20% drop and finally one neuron in the output.
# 
# I will run 20 epochs with a batch size of 100,000. I want to explore this numbers as I don't know how this will impact the prediction.

# In[13]:


model = Sequential()
model.add(LSTM(64, input_shape=(X_train2.shape[1],X_train2.shape[2])))
model.add(Dropout(.1))
model.add(Dense(32))
model.add(Dropout(.2))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(X_train2, y_train2, batch_size = 100000, epochs = 20, verbose=2,
    validation_data=(X_val2,y_val2))


# It took quite long...

# Now I prepare the test with the two previous months data for prediction.

# In[14]:


df_test = pd.read_csv(
        '../input/test.csv',
        ).set_index(['shop_id', 'item_id'])


# In[15]:


X_test = df_test
X = df_t.iloc[:, 32:34]
X_test = X_test.join(X)
print (X_test)


# In[16]:


X_test = X_test.fillna(0)
print (X_test)


# In[18]:


X_test2 = X_test.iloc[:, 1:3]
X_test2 = X_test2.as_matrix()
print (X_test2.shape)


# Same reshape to 3D.

# In[19]:


X_test2 = X_test2.reshape(X_test2.shape[0], 1, X_test2.shape[1])
predict = model.predict(X_test2)


# And submission.

# In[20]:


df_test['item_cnt_month'] = predict
df_test.to_csv('accountant_keras1.csv', float_format = '%.2f', index=None)

