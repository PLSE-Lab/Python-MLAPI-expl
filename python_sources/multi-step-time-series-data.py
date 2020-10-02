#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Dependencies

import numpy as np
from numpy import nan
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense


# In[ ]:


data = pd.read_csv('../input/power-consumption-of-house/power_consumption_of_house.txt', sep = ';', parse_dates = True, low_memory = False)

data['date_time'] = data['Date'].str.cat(data['Time'], sep= ' ')
data.drop(['Date', 'Time'], inplace= True, axis = 1)

data.set_index(['date_time'], inplace=True)
data.replace('?', nan, inplace=True)
data = data.astype('float')
data.head()


# In[ ]:


#First check how many values are null
np.isnan(data).sum()

#fill the null value

def fill_missing(data):
    one_day = 24*60
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if np.isnan(data[row, col]):
                data[row, col] = data[row-one_day, col]

fill_missing(data.values)

#Again check the data after filling the value
np.isnan(data).sum()


# In[ ]:


data.describe()
data.shape


# In[ ]:


data.head()


# In[ ]:


# Converting the index as date
data.index = pd.to_datetime(data.index)


# In[ ]:


data = data.resample('D').sum()


# In[ ]:


data.head()


# In[ ]:



fig, ax = plt.subplots(figsize=(18,18))

for i in range(len(data.columns)):
    plt.subplot(len(data.columns), 1, i+1)
    name = data.columns[i]
    plt.plot(data[name])
    plt.title(name, y=0, loc = 'right')
    plt.yticks([])
plt.show()
fig.tight_layout()


# # Exploring Active power consumption for each year

# In[ ]:


years = ['2007', '2008', '2009', '2010']

fig, ax = plt.subplots(figsize=(18,18))
for i in range(len(years)):
    plt.subplot(len(years), 1, i+1)
    year = years[i]
    active_power_data = data[str(year)]
    plt.plot(active_power_data['Global_active_power'])
    plt.title(str(year), y = 0, loc = 'left')
plt.show()
fig.tight_layout()


# # Power consumption distribution with histogram

# In[ ]:


fig, ax = plt.subplots(figsize=(18,18))

for i in range(len(years)):
    plt.subplot(len(years), 1, i+1)
    year = years[i]
    active_power_data = data[str(year)]
    active_power_data['Global_active_power'].hist(bins = 200)
    plt.title(str(year), y = 0, loc = 'left')
plt.show()
fig.tight_layout()


# In[ ]:


# for full data

fig, ax = plt.subplots(figsize=(18,18))

for i in range(len(data.columns)):
    plt.subplot(len(data.columns), 1, i+1)
    name = data.columns[i]
    data[name].hist(bins=200)
    plt.title(name, y=0, loc = 'right')
    plt.yticks([])
plt.show()
fig.tight_layout()


# ## What can we predict
# 
# Forecast hourly consumption for the next day.  
# Forecast daily consumption for the next week.  
# Forecast daily consumption for the next month.    
# Forecast monthly consumption for the next year.  

# ## Modeling Methods  
# There are many modeling methods and few of those are as follows
# 
# Naive Methods -> Naive methods would include methods that make very simple, but often very effective assumptions.  
# Classical Linear Methods -> Classical linear methods include techniques are very effective for univariate time series forecasting  
# Machine Learning Methods -> Machine learning methods require that the problem be framed as a supervised learning problem.  
# k-nearest neighbors.  
# SVM  
# Decision trees  
# Random forest  
# Gradient boosting machines  
# Deep Learning Methods -> combinations of CNN LSTM and ConvLSTM, have proven effective on time series classification tasks  
# CNN  
# LSTM  
# CNN - LSTM  

# In[ ]:


data_train = data.loc[:'2009-12-31', :]['Global_active_power']
data_train.head()


# In[ ]:


data_test = data['2010']['Global_active_power']
data_test.head()


# In[ ]:


data_train.shape


# In[ ]:


data_test.shape


# # Prepare Training data

# In[ ]:


data_train = np.array(data_train)
print(data_train)

X_train, y_train = [], []
for i in range(7, len(data_train)-7):
    X_train.append(data_train[i-7:i])
    y_train.append(data_train[i:i+7])
    
X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape, y_train.shape


# In[ ]:


pd.DataFrame(X_train).head()


# In[ ]:


x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(X_train)

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)


# In[ ]:


X_train = X_train.reshape(1098, 7, 1)


# In[ ]:


X_train.shape


# # Build LSTM Network

# In[ ]:


model = Sequential()
model.add(LSTM(units = 200, activation = 'relu', input_shape=(7,1)))
model.add(Dense(7))

model.compile(loss='mse', optimizer='adam')


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train, y_train, epochs = 100)


# # Prepare test dataset and test LSTM model

# In[ ]:


data_test = np.array(data_test)


# In[ ]:


X_test, y_test = [], []

for i in range(7, len(data_test)-7):
    X_test.append(data_test[i-7:i])
    y_test.append(data_test[i:i+7])


# In[ ]:


X_test, y_test = np.array(X_test), np.array(y_test)


# In[ ]:



X_test = x_scaler.transform(X_test)
y_test = y_scaler.transform(y_test)


# In[ ]:


X_test = X_test.reshape(331,7,1)


# In[ ]:


X_test.shape


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


y_pred = y_scaler.inverse_transform(y_pred)
y_pred


# In[ ]:


y_true = y_scaler.inverse_transform(y_test)
y_true


# # Evaluate the Model

# In[ ]:



def evaluate_model(y_true, y_predicted):
    scores = []
    
    #calculate scores for each day
    for i in range(y_true.shape[1]):
        mse = mean_squared_error(y_true[:, i], y_predicted[:, i])
        rmse = np.sqrt(mse)
        scores.append(rmse)
    
    #calculate score for whole prediction
    total_score = 0
    for row in range(y_true.shape[0]):
        for col in range(y_predicted.shape[1]):
            total_score = total_score + (y_true[row, col] - y_predicted[row, col])**2
    total_score = np.sqrt(total_score/(y_true.shape[0]*y_predicted.shape[1]))
    
    return total_score, scores


# In[ ]:


evaluate_model(y_true, y_pred)

