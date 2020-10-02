#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/brent-oil-prices/BrentOilPrices.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by = ['Date'])
data.isna().sum()


# In[ ]:


data.index = data['Date']
data = data.drop(['Date'],axis=1)
data = data[6000:]


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(data)
plt.title('Graph')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[ ]:


sc = MinMaxScaler(feature_range = (0, 1))
data = sc.fit_transform(data)


# In[ ]:


train_data, test_data = data[0:int(len(data)*0.8), :], data[int(len(data)*0.8):len(data), :]


# In[ ]:


def prepre_dataset( dataset, window ):
    data, label =[],[]
    for i in range( len(dataset) - window - 1 ):
        d = dataset[ i:( i+window ), 0 ]
        data.append(d)
        label.append(dataset[i+window,0])
    return np.array(data), np.array(label)


# In[ ]:


train_x, train_y = prepre_dataset(train_data,90)
test_x, test_y = prepre_dataset(test_data,90)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1],1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1],1))


# In[ ]:


test_y.shape


# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units = 60, input_shape = (train_x.shape[1], 1)))
model.add(tf.keras.layers.Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[ ]:


history = model.fit(train_x, train_y, epochs = 10, batch_size = 15, validation_data = (test_x, test_y), verbose =1)


# In[ ]:


test_predict = model.predict(test_x)


# In[ ]:


test_predict = sc.inverse_transform(test_predict)
test_y = sc.inverse_transform([test_y])


# In[ ]:


aa=[x for x in range(180)]
plt.figure(figsize=(8,4))
plt.plot(aa, test_y[0][:180], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:180], 'r', label="prediction")
plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

