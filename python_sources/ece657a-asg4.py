#!/usr/bin/env python
# coding: utf-8

# # Assignment 4 - Coronavirus
# 
# Created by _Waterloo Warriors_
# 
# Mohammad Dib
# 
# Daniel Weber

# This kernel is created to face some of the current challenges about the Coronavirus related to time series data and machine learning. It provides some basic visualizations and compares different deep learning approaches to predict the confirmed cases caused by COVID-19. In detail, Long Short Term Memory (LSTM) and Gated Recurrent Unit (GRU) models are considered.

# ## Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
plt.style.use('fivethirtyeight')
sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)

# Prediction tasks
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Data import

# In[ ]:


# Global confirmed cases
confirmed_file = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv"
confirmed = pd.read_csv(confirmed_file)

# Global deaths
deaths_file = "../input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv"
deaths = pd.read_csv(deaths_file)

# Global recovered
recovered_file = "../input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv"
recovered = pd.read_csv(recovered_file)


# In[ ]:


confirmed.head()


# In[ ]:


deaths.head()


# In[ ]:


recovered.head()


# ## Some visualization

# In[ ]:


# Some functions to help out with
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real Confirmed')
    plt.plot(predicted, color='blue',label='Predicted Confirmed')
    plt.title('Confirmed Cases')
    plt.xlabel('Time')
    plt.ylabel('Confirmed Cases')
    plt.legend()
    plt.show()

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))


# In[ ]:


countries=['Brazil', 'Canada', 'Germany']
y = confirmed.loc[confirmed['Country/Region']=='Italy'].iloc[0,4:]
s = pd.DataFrame({'Italy':y})
for c in countries:    
    #pyplot.plot(range(y.shape[0]),y,'r--')
    s[c] = confirmed.loc[confirmed['Country/Region']==c].iloc[0,4:]
#pyplot.plot(range(y.shape[0]),y,'g-')
s.plot.line()


# In[ ]:


country = 'Italy'
y1 = confirmed.loc[confirmed['Country/Region']==country].iloc[0,4:]
y2 = deaths.loc[confirmed['Country/Region']==country].iloc[0,4:]
y3 = recovered.loc[confirmed['Country/Region']==country].iloc[0,4:]

df = pd.DataFrame({'Confirmed':y1, 'Deaths':y2, 'Recovered':y3})
df.plot.line()


# ## Predictions of confirmed cases using LSTM
# 
# According to https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru, https://www.kaggle.com/vanshjatana/machine-learning-on-coronavirus and https://www.kaggle.com/azizovitic/sarima-lstm-for-novel-corona-virus-2019-dataset
# 
# ### Confirmed Cases in Italy

# In[ ]:


# choose country and build data
country = 'Italy'

data = confirmed.loc[confirmed['Country/Region']==country].iloc[0,4:]
dti = pd.date_range('2020-01-22', periods=data.shape[0], freq='D')
data = pd.DataFrame(data=data.values,columns=['Confirmed'],index=dti)


# There will always be a number of 5 days to be predicted.

# In[ ]:


# do train/test split
n_days_toPredict = 5

train_data = data[:len(data)-n_days_toPredict]
test_data = data[len(data)-n_days_toPredict:]


# In[ ]:


# plot train/test data
data['Confirmed'][:train_data.index[-1]].plot(figsize=(16,4),legend=True)
data['Confirmed'][train_data.index[-1]:].plot(figsize=(16,4),legend=True)
plt.legend(['Training set','Test set'])
plt.title('Confirmed Cases')
plt.show()


# In[ ]:


# LSTM architechture

scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
n_input = 7
n_features = 1
                             
generator = TimeseriesGenerator(scaled_train_data,scaled_train_data, length=n_input, batch_size=1)

lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units = 1))
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm_model.fit(generator, epochs = 30)


# In[ ]:


# plot loss 
losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize = (30,4))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0,100,1))
plt.plot(range(len(losses_lstm)), losses_lstm)


# In[ ]:


# do predictions 
lstm_predictions_scaled = []

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# In[ ]:


# transform predictions
prediction = pd.DataFrame(scaler.inverse_transform(lstm_predictions_scaled),columns=['PREDICTED Confirmed'],index=test_data.index)


# In[ ]:


return_rmse(test_data,prediction)


# In[ ]:


plot_predictions(test_data,prediction)


# In[ ]:


plt.plot(train_data, label='Historical Confirmed')
plt.plot(prediction, label='Predicted Confirmed')
plt.plot(test_data, label='Actual Confirmed')
plt.legend();


# ### Global Confirmed Cases

# In[ ]:


global_cases = confirmed.iloc[:,4:].sum(axis=0)
dti = pd.date_range('2020-01-22', periods=data.shape[0], freq='D')
data2 = pd.DataFrame(data=global_cases.values,columns=['Confirmed'],index=dti)


# In[ ]:


# do train/test split
n_days_toPredict = 5

train_dataG = data2[:len(data2)-n_days_toPredict]
test_dataG = data2[len(data2)-n_days_toPredict:]


# In[ ]:


# plot train/test data
data2['Confirmed'][:train_dataG.index[-1]].plot(figsize=(16,4),legend=True)
data2['Confirmed'][train_dataG.index[-1]:].plot(figsize=(16,4),legend=True)
plt.legend(['Training set','Test set'])
plt.title('Confirmed Cases')
plt.show()


# In[ ]:


# LSTM architechture

scaler = MinMaxScaler()
scaler.fit(train_dataG)
scaled_train_dataG = scaler.transform(train_dataG)
scaled_test_dataG = scaler.transform(test_dataG)
n_input = 7
n_features = 1
                             
generator = TimeseriesGenerator(scaled_train_dataG,scaled_train_dataG, length=n_input, batch_size=1)

lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units = 1))
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm_model.fit(generator, epochs = 30)


# In[ ]:


# plot loss 
losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize = (30,4))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0,100,1))
plt.plot(range(len(losses_lstm)), losses_lstm)


# In[ ]:


# do predictions 
lstm_predictions_scaledG = []

batch = scaled_train_dataG[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_dataG)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaledG.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# In[ ]:


# transform predictions
prediction_global = pd.DataFrame(scaler.inverse_transform(lstm_predictions_scaledG),columns=['PREDICTED Confirmed'],index=test_dataG.index)


# In[ ]:


return_rmse(test_dataG,prediction_global)


# In[ ]:


plot_predictions(test_dataG,prediction_global)


# In[ ]:


plt.plot(train_dataG, label='Historical Confirmed')
plt.plot(prediction_global, label='Predicted Confirmed')
plt.plot(test_dataG, label='Actual Confirmed')
plt.legend();


# The LSTM model can achieve reasonable accuracy, but differs significantly between several runs. It tends to show a more flattening effect than the actual development. Also, the global data tends to perform better on several runs.

# ### Using the whole dataset and doing the forecast for global cases
# 
# **CAUTION:** Update daterange index for predicitons, if data is updated before.

# In[ ]:


# forecast length
forecast_length = 5

train_data_forecast = data2

# LSTM architechture

scaler = MinMaxScaler()
scaler.fit(train_data_forecast)
scaled_train_data_forecast = scaler.transform(train_data_forecast)
n_input = 7
n_features = 1
                             
generator = TimeseriesGenerator(scaled_train_data_forecast,scaled_train_data_forecast, length=n_input, batch_size=1)

lstm_model = Sequential()
lstm_model.add(LSTM(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50, return_sequences = True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units = 50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units = 1))
lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm_model.fit(generator, epochs = 30)

# predictions

lstm_predictions_scaled_forecast = []

batch = scaled_train_data_forecast[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(forecast_length):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled_forecast.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
    
prediction_forecast = pd.DataFrame(scaler.inverse_transform(lstm_predictions_scaled_forecast),columns=['PREDICTED Confirmed'],index=pd.date_range('2020-04-18', periods=forecast_length, freq='D'))


# In[ ]:


plt.plot(train_data_forecast, label='Historical Confirmed')
plt.plot(prediction_forecast, label='Predicted Confirmed')
plt.legend();


# ## Predictions of confirmed cases using GRU
# 

# ### Italy

# In[ ]:


# choose country and build data
country = 'Italy'

data = confirmed.loc[confirmed['Country/Region']==country].iloc[0,4:]
dti = pd.date_range('2020-01-22', periods=data.shape[0], freq='D')
data = pd.DataFrame(data=data.values,columns=['Confirmed'],index=dti)


# There will always be a number of 5 days to be predicted.

# In[ ]:


# do train/test split
n_days_toPredict = 5

train_data = data[:len(data)-n_days_toPredict]
test_data = data[len(data)-n_days_toPredict:]


# In[ ]:


# plot train/test data
data['Confirmed'][:train_data.index[-1]].plot(figsize=(16,4),legend=True)
data['Confirmed'][train_data.index[-1]:].plot(figsize=(16,4),legend=True)
plt.legend(['Training set','Test set'])
plt.title('Confirmed Cases')
plt.show()


# In[ ]:


# GRU architechture

scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)
n_input = 7
n_features = 1
                             
generator = TimeseriesGenerator(scaled_train_data,scaled_train_data, length=n_input, batch_size=1)

gru_model = Sequential()
gru_model.add(GRU(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(units = 50, return_sequences = True))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(units = 50))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(units = 1))
gru_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
gru_model.fit(generator, epochs = 30)


# In[ ]:


# plot loss 
losses_gru = gru_model.history.history['loss']
plt.figure(figsize = (30,4))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0,100,1))
plt.plot(range(len(losses_gru)), losses_gru)


# In[ ]:


# do predictions 
gru_predictions_scaled = []

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    gru_pred = gru_model.predict(current_batch)[0]
    gru_predictions_scaled.append(gru_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[gru_pred]],axis=1)


# In[ ]:


# transform predictions
prediction_gru = pd.DataFrame(scaler.inverse_transform(gru_predictions_scaled),columns=['PREDICTED Confirmed'],index=test_data.index)


# In[ ]:


return_rmse(test_data,prediction_gru)


# In[ ]:


plot_predictions(test_data,prediction_gru)


# In[ ]:


plt.plot(train_data, label='Historical Confirmed')
plt.plot(prediction_gru, label='Predicted Confirmed')
plt.plot(test_data, label='Actual Confirmed')
plt.legend();


# ### Global Confirmed Cases

# In[ ]:


global_cases = confirmed.iloc[:,4:].sum(axis=0)
dti = pd.date_range('2020-01-22', periods=data.shape[0], freq='D')
data2 = pd.DataFrame(data=global_cases.values,columns=['Confirmed'],index=dti)


# In[ ]:


# do train/test split
n_days_toPredict = 5

train_dataG = data2[:len(data2)-n_days_toPredict]
test_dataG = data2[len(data2)-n_days_toPredict:]


# In[ ]:


# plot train/test data
data2['Confirmed'][:train_dataG.index[-1]].plot(figsize=(16,4),legend=True)
data2['Confirmed'][train_dataG.index[-1]:].plot(figsize=(16,4),legend=True)
plt.legend(['Training set','Test set'])
plt.title('Confirmed Cases')
plt.show()


# In[ ]:


# GRU architechture

scaler = MinMaxScaler()
scaler.fit(train_dataG)
scaled_train_dataG = scaler.transform(train_dataG)
scaled_test_dataG = scaler.transform(test_dataG)
n_input = 7
n_features = 1
                             
generator = TimeseriesGenerator(scaled_train_dataG,scaled_train_dataG, length=n_input, batch_size=1)

gru_model = Sequential()
gru_model.add(GRU(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(units = 50, return_sequences = True))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(units = 50))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(units = 1))
gru_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
gru_model.fit(generator, epochs = 30)


# In[ ]:


# plot loss 
losses_gru = gru_model.history.history['loss']
plt.figure(figsize = (30,4))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0,100,1))
plt.plot(range(len(losses_gru)), losses_gru)


# In[ ]:


# do predictions 
gru_predictions_scaledG = []

batch = scaled_train_dataG[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_dataG)):   
    gru_pred = gru_model.predict(current_batch)[0]
    gru_predictions_scaledG.append(gru_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[gru_pred]],axis=1)


# In[ ]:


# transform predictions
prediction_global_gru = pd.DataFrame(scaler.inverse_transform(gru_predictions_scaledG),columns=['PREDICTED Confirmed'],index=test_dataG.index)


# In[ ]:


return_rmse(test_dataG,prediction_global_gru)


# In[ ]:


plot_predictions(test_dataG,prediction_global_gru)


# In[ ]:


plt.plot(train_dataG, label='Historical Confirmed')
plt.plot(prediction_global_gru, label='Predicted Confirmed')
plt.plot(test_dataG, label='Actual Confirmed')
plt.legend();


# The GRU model achieved high accuracy, also the global data performed better.

# ## Using the whole dataset and doing the forecast for global cases
# 
# **CAUTION:** Update daterange index for predicitons, if data is updated before.

# In[ ]:


# forecast length
forecast_length = 5

train_data_forecast = data2

# GRU architechture

scaler = MinMaxScaler()
scaler.fit(train_data_forecast)
scaled_train_data_forecast = scaler.transform(train_data_forecast)
n_input = 7
n_features = 1
                             
generator = TimeseriesGenerator(scaled_train_data_forecast,scaled_train_data_forecast, length=n_input, batch_size=1)

gru_model = Sequential()
gru_model.add(GRU(units = 50, return_sequences = True, input_shape = (n_input, n_features)))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(units = 50, return_sequences = True))
gru_model.add(Dropout(0.2))
gru_model.add(GRU(units = 50))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(units = 1))
gru_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
gru_model.fit(generator, epochs = 30)

# predictions

gru_predictions_scaled_forecast = []

batch = scaled_train_data_forecast[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(forecast_length):   
    gru_pred = gru_model.predict(current_batch)[0]
    gru_predictions_scaled_forecast.append(gru_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[gru_pred]],axis=1)
    
prediction_forecast_gru = pd.DataFrame(scaler.inverse_transform(gru_predictions_scaled_forecast),columns=['PREDICTED Confirmed'],index=pd.date_range('2020-04-18', periods=forecast_length, freq='D'))


# In[ ]:


plt.plot(train_data_forecast, label='Historical Confirmed')
plt.plot(prediction_forecast_gru, label='Predicted Confirmed')
plt.legend();


# ## Comparison

# ### Visual Comparison

# In[ ]:


# Italy
plt.plot(train_data['2020-03-22':], label='Historical Confirmed')
plt.plot(prediction, label='Predicted Confirmed by LSTM')
plt.plot(prediction_gru, label='Predicted Confirmed by GRU')
plt.plot(test_data, label='Actual Confirmed')
plt.title("Italy")
plt.legend();


# In[ ]:


# global
plt.plot(train_dataG['2020-03-22':], label='Historical Confirmed')
plt.plot(prediction_global, label='Predicted Confirmed by LSTM')
plt.plot(prediction_global_gru, label='Predicted Confirmed by GRU')
plt.plot(test_dataG, label='Actual Confirmed')
plt.title("Global")
plt.legend();


# ### Numerical Comparison

# In[ ]:


results = pd.DataFrame(data=np.zeros((2,4)),columns=['RMSE Italy LSTM','RMSE Italy GRU','RMSE Global LSTM','RMSE Globale GRU'],index=['Absolute RMSE','Relative RMSE'])
# Absolute RMSE
# LSTM
results.iloc[0,0] = resultsrmse_Italy_lstm = math.sqrt(mean_squared_error(prediction,test_data))
results.iloc[0,2] = math.sqrt(mean_squared_error(prediction_global,test_dataG))
# GRU
results.iloc[0,1] = resultsrmse_Italy_gru = math.sqrt(mean_squared_error(prediction_gru,test_data))
results.iloc[0,3] = math.sqrt(mean_squared_error(prediction_global_gru,test_dataG))

# LSTM
results.iloc[1,0] = results.iloc[0,0] / test_data.mean().values[0]
results.iloc[1,2] = results.iloc[0,2] / test_dataG.mean().values[0]
# GRU
results.iloc[1,1] = results.iloc[0,1] / test_data.mean().values[0]
results.iloc[1,3] = results.iloc[0,3] / test_dataG.mean().values[0]

results


# The results in the table above shows that in both cases (Italy and Global) the GRU model gave better results. Additionally, it is worth noting the best results, by a significent margin, were achieved using the global data and the GRU model.     

# In[ ]:




