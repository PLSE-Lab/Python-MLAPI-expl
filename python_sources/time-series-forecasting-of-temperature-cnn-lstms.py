#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Dense,RepeatVector, LSTM, Dropout
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model


# In[ ]:


df = pd.read_csv("/kaggle/input/delhi-weather-data/testset.csv")


# In[ ]:


df.head()


# # Part 1: A quick analysis of Weather in Delhi

# In[ ]:


df[' _conds'].value_counts()


# In[ ]:


plt.figure(figsize=(15,10))
df[' _conds'].value_counts().head(15).plot(kind='bar')

plt.title('15 most common weathers in Delhi')
plt.show()


# **Haze and Smoke are most common weatehrs conditions in Delhi**

# In[ ]:


plt.figure(figsize=(15, 10))
plt.title("Common wind direction in delhi")
df[' _wdire'].value_counts().plot(kind="bar")
plt.plot()


# **
# North and West are the most common wind directions in dehi.**

# In[ ]:


plt.figure(figsize=(15, 10))
sns.distplot(df[' _tempm'],bins=[i for i in range(0,61,5)], kde=False)
plt.title("Distribution of Temperatures")
plt.grid()
plt.show()


# **Most common temperature scale in Delhi is from 25 to 35 degree.**

# In[ ]:


df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])


# In[ ]:


df['datetime_utc']


# In[ ]:


# imputing the missing value in temperatre feature with mean.
df[' _tempm'].fillna(df[' _tempm'].mean(), inplace=True)


# In[ ]:


df[' _tempm'].isna().sum()
# filled all missing values with mean()


# In[ ]:


str(df['datetime_utc'][0])


# In[ ]:


# a function to extract year part from the whole date
def get_year(x):
  return x[0:4]


# In[ ]:


# a function to extract month part from the whole date
def get_month(x):
  return x[5:7]


# In[ ]:


# making two new features year and month
df['year'] = df['datetime_utc'].apply(lambda x: get_year(str(x)))
df['month'] = df['datetime_utc'].apply(lambda x: get_month(str(x)))


# In[ ]:


df['year']


# In[ ]:


temp_year = pd.crosstab(df['year'], df['month'], values=df[' _tempm'], aggfunc='mean')


# In[ ]:


plt.figure(figsize=(15, 10))
sns.heatmap(temp_year, cmap='coolwarm', annot=True)
plt.title("Average Tempearture in Delhi from 1996 to 2017")
plt.show()


# In[ ]:


df[' _hum'].isna().sum()


# In[ ]:


# imputing missing values in _hum feature with mean
df[' _hum'].fillna(df[' _hum'].mean(), inplace=True)


# In[ ]:


humidity_year = pd.crosstab(df['year'], df['month'], values=df[' _hum'], aggfunc='mean')


# In[ ]:


plt.figure(figsize=(15, 10))
sns.heatmap(humidity_year, cmap='coolwarm', annot=True)
plt.title("Average Humidity in Delhi from 1996 to 2017")
plt.show()


# # Part 2: Time Series Forecasting

# In[ ]:


# taking only temperature feature as values and datetime feature as index in the dataframe for time series forecasting of temperature
data = pd.DataFrame(list(df[' _tempm']), index=df['datetime_utc'], columns=['temp'])


# In[ ]:


data


# In[ ]:


# resampling data with date frequency for time series forecasting
data = data.resample('D').mean()


# In[ ]:


data.temp.isna().sum()


# In[ ]:


data.fillna(data['temp'].mean(), inplace=True)


# In[ ]:


data.temp.isna().sum()


# In[ ]:


data.shape


# In[ ]:


data


# In[ ]:


plt.figure(figsize=(25, 7))
plt.plot(data, linewidth=.5)
plt.grid()
plt.title("Time Series (Years vs Temp.)")
plt.show()


# In[ ]:


# Scaling data to get rid of outliers
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(-1,1))
data_scaled = scalar.fit_transform(data)


# In[ ]:


data_scaled


# In[ ]:


data_scaled.shape


# In[ ]:


steps = 30
inp = []
out = []
for i in range(len(data_scaled)- (steps)):
    inp.append(data_scaled[i:i+steps])
    out.append(data_scaled[i+steps])


# In[ ]:


inp=np.asanyarray(inp)
out=np.asanyarray(out)


# In[ ]:


x_train = inp[:7300,:,:]
x_test = inp[7300:,:,:]    
y_train = out[:7300]    
y_test= out[7300:]


# In[ ]:


inp.shape


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
early_stop = EarlyStopping(monitor = "loss", mode = "min", patience = 7)
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(30,1)))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(30))
model.add(LSTM(units=100, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True, activation='relu'))
model.add(LSTM(units=100, return_sequences=True, activation='relu'))
model.add(Bidirectional(LSTM(128, activation='relu')))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


# In[ ]:


plot_model(model, to_file='model.png')


# In[ ]:


history = model.fit(x_train,y_train,epochs=300, verbose=1, callbacks = [early_stop] )


# In[ ]:


model.save("./regressor.hdf5")


# In[ ]:


predict = model.predict(x_test)


# In[ ]:


predict = scalar.inverse_transform(predict)


# In[ ]:


Ytesting = scalar.inverse_transform(y_test)


# In[ ]:


plt.figure(figsize=(20,9))
plt.plot(Ytesting , 'blue', linewidth=5)
plt.plot(predict,'r' , linewidth=4)
plt.legend(('Test','Predicted'))
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(Ytesting, predict)


# This is a demonstration of using CNN-LSTMs for Time Series Forecasting. We can also improve the model to make better predictions.
# If you have any suggestion, please comment.
