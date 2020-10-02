#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/for-simple-exercises-time-series-forecasting/Alcohol_Sales.csv', index_col = 'DATE', parse_dates=True)
df.index.freq = 'MS'
df.head()


# In[ ]:


# Renaming the only column present to Sales
df.columns = ['Sales']


# In[ ]:


df.plot(figsize=(12,8))


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[ ]:


# Decompose the Sales data
results = seasonal_decompose(df['Sales'])


# In[ ]:


results.plot();


# In[ ]:


results.seasonal.plot(figsize=(10,6))


# In[ ]:


len(df)


# In[ ]:


# We'll predict sales for next one year, as it is monthly data, so we'll have test size of 12 and train size of len(df) - 12
# Splitting the data into train and test

train = df.iloc[:313]
test = df.iloc[313:]


# In[ ]:


len(test)


# In[ ]:


# We need to scale or normalize the data to pass it into RNN (recurrent neural network)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[ ]:


# fit the training data
scaler.fit(train) # finds the max value in train data


# In[ ]:


# transform the training data
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[ ]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[ ]:


# Create TimeseriesGenerator object for training dataset (define how long the trainig sequence to be (n_input) and the very next point to predict)
n_input = 12
n_features = 1 # number of columns in your dataset

train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# In the above n_feature should be decided based on seasonality, if seasonality is based over the year, then as it is monthly dataset, so, n_feature should be at least 12, in order for RNN to pickup at least seasonality

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[ ]:


model = Sequential()

model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[ ]:


model.summary()


# In[ ]:


model.fit_generator(train_generator, epochs=25)


# In[ ]:


model.history.history.keys()


# In[ ]:


# plotting loss versus their range of values
myloss = model.history.history['loss']
plt.plot(range(len(myloss)), myloss)


# In[ ]:


# For predicting the first data in test set we need the last 12 data points from the test dataset
first_eval_batch = scaled_train[-12:]


# In[ ]:


first_eval_batch


# In[ ]:


first_eval_batch.shape


# In[ ]:


# reshaping first_eval_batch as the input which is being passed to TimeseriesGenerator must be 3D (batch_size, n_input, n_features) 
first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))


# In[ ]:


model.predict(first_eval_batch)


# #### Forecast using RNN model

# In[ ]:


# holding my predictions
test_predictions = [] 

# last n_input points from training set
first_eval_batch = scaled_train[-n_input:]
# Reshape this to the format RNN wants (same as TimeseriesGenerator)
current_batch = first_eval_batch.reshape((1, n_input, n_features))


# In[ ]:


# how far into future will I forecast
for i in range(len(test)):
    # one timestep ahead of historical 12 points
    current_pred = model.predict(current_batch)[0]
    # store that prediction
    test_predictions.append(current_pred)
    # update the current batch to include prediction
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)


# In[ ]:


len(test_predictions)


# In[ ]:


# Performing reverse scaling on the test predictions
true_predictions = scaler.inverse_transform(test_predictions)


# In[ ]:


true_predictions


# In[ ]:


# Add prediction column to dataset
test['Predictions'] = true_predictions


# In[ ]:


# Plotting original sales with predicted sales
test.plot(figsize=(12,8))


# In[ ]:


# Saving the trained model for future reference
model.save('timeseriesmodel.h5')


# In[ ]:


from keras.models import load_model


# In[ ]:


# loading the model
timeseries_rnn_model = load_model('timeseriesmodel.h5')


# In[ ]:


timeseries_rnn_model.summary()

