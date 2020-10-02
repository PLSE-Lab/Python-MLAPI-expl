#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import keras
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# reading csv file
ds = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv')
ds


# In[ ]:


len(ds)   # length of train dataset


# In[ ]:


# reading test dataset
tds = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTest.csv')
tds.head()


# In[ ]:


len(tds)     # length of test dataset


# # Appending train and test for scaling

# In[ ]:


ds = ds.append(tds, ignore_index=True)
ds.shape


# In[ ]:


ds = ds['meantemp']
ds = np.array(ds)
ds = ds.reshape(-1,1)
ds


# # Visualizing the dataset

# In[ ]:


plt.plot(ds)


# # Standardization

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

ds = scaler.fit_transform(ds)
ds


# In[ ]:


scaler.scale_


# # spliting training & testing data

# In[ ]:


train = ds[0:1462]
test = ds[1462:]
train.shape , test.shape


# In[ ]:


def get_data(dataset, look_back):
  datax = []
  datay = []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    datax.append(a)
    datay.append(dataset[i+look_back, 0])
  return np.array(datax), np.array(datay)


# In[ ]:


look_back = 1
x_train, y_train = get_data(train, look_back)
x_test, y_test = get_data(test, look_back)


# In[ ]:


x_train , y_train


# # Reshaping

# In[ ]:


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

x_train.shape , x_test.shape


# # **LSTM**

# In[ ]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(16, activation='tanh', input_shape = (1,1)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss = 'mean_squared_error') 
model.fit(x_train, y_train, epochs = 20, batch_size=1)


# # Predicting test data

# In[ ]:


y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_pred


# In[ ]:


y_test = np.array(y_test)
y_test = y_test.reshape(-1,1)
y_test = scaler.inverse_transform(y_test)


# # plotting predicted & test labels

# In[ ]:


plt.plot(y_test, color='blue', label = 'Actual Values')
plt.plot(y_pred, color='brown', label = 'Predicted Values')
plt.ylabel('Passengers')
plt.legend()

