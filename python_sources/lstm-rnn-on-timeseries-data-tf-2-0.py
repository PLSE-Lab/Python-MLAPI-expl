#!/usr/bin/env python
# coding: utf-8

# # RNN and LSTM Example on Sine Wave

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Data
# 
# Let's use Numpy to create a simple sine wave.

# In[ ]:


x = np.linspace(0,50,501)
y = np.sin(x)


# In[ ]:


x


# In[ ]:


y


# In[ ]:


plt.plot(x,y)


# Let's turn this into a DataFrame

# In[ ]:


df = pd.DataFrame(data=y,index=x,columns=['Sine'])


# In[ ]:


df


# ## Train Test Split
# 
# Note! This is very different from our usual test/train split methodology!

# In[ ]:


len(df)


# In[ ]:


test_percent = 0.1


# In[ ]:


len(df)*test_percent


# In[ ]:


test_point = np.round(len(df)*test_percent)


# In[ ]:


test_ind = int(len(df) - test_point)


# In[ ]:


test_ind


# In[ ]:


train = df.iloc[:test_ind]
test = df.iloc[test_ind:]


# In[ ]:


train


# In[ ]:


test


# ## Scale Data

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


# IGNORE WARNING ITS JUST CONVERTING TO FLOATS
# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET
scaler.fit(train)


# In[ ]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# # Time Series Generator
# 
# This class takes in a sequence of data-points gathered at
# equal intervals, along with time series parameters such as
# stride, length of history, etc., to produce batches for
# training/validation.
# 
# #### Arguments
#     data: Indexable generator (such as list or Numpy array)
#         containing consecutive data points (timesteps).
#         The data should be at 2D, and axis 0 is expected
#         to be the time dimension.
#     targets: Targets corresponding to timesteps in `data`.
#         It should have same length as `data`.
#     length: Length of the output sequences (in number of timesteps).
#     sampling_rate: Period between successive individual timesteps
#         within sequences. For rate `r`, timesteps
#         `data[i]`, `data[i-r]`, ... `data[i - length]`
#         are used for create a sample sequence.
#     stride: Period between successive output sequences.
#         For stride `s`, consecutive output samples would
#         be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
#     start_index: Data points earlier than `start_index` will not be used
#         in the output sequences. This is useful to reserve part of the
#         data for test or validation.
#     end_index: Data points later than `end_index` will not be used
#         in the output sequences. This is useful to reserve part of the
#         data for test or validation.
#     shuffle: Whether to shuffle output samples,
#         or instead draw them in chronological order.
#     reverse: Boolean: if `true`, timesteps in each output sample will be
#         in reverse chronological order.
#     batch_size: Number of timeseries samples in each batch
#         (except maybe the last one).

# In[ ]:


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[ ]:


# scaled_train


# In[ ]:


# define generator
length = 2 # Length of the output sequences (in number of timesteps)
batch_size = 1 #Number of timeseries samples in each batch
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)


# In[ ]:


len(scaled_train)


# In[ ]:


len(generator) # n_input = 2


# In[ ]:


# scaled_train


# In[ ]:


# What does the first batch look like?
X,y = generator[0]


# In[ ]:


print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# In[ ]:


# Let's redefine to get 10 steps back and then predict the next step out
length = 10 # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)


# In[ ]:


# What does the first batch look like?
X,y = generator[0]


# In[ ]:


print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# In[ ]:


length = 50 # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)


# Now you will be able to edit the length so that it makes sense for your time series!

# ### Create the Model

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,SimpleRNN


# In[ ]:


# We're only using one feature in our time series
n_features = 1


# In[ ]:


# define model
model = Sequential()

# Simple RNN layer
model.add(SimpleRNN(50,input_shape=(length, n_features)))

# Final Prediction
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# In[ ]:


model.summary()


# In[ ]:


# fit model
model.fit_generator(generator,epochs=5)


# In[ ]:


model.history.history.keys()


# In[ ]:


losses = pd.DataFrame(model.history.history)
losses.plot()


# ## Evaluate on Test Data

# In[ ]:


first_eval_batch = scaled_train[-length:]


# In[ ]:


first_eval_batch


# In[ ]:


first_eval_batch = first_eval_batch.reshape((1, length, n_features))


# In[ ]:


model.predict(first_eval_batch)


# In[ ]:


scaled_test[0]


# Now let's put this logic in a for loop to predict into the future for the entire test range.
# 
# ----

# In[ ]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))


# In[ ]:


current_batch.shape


# In[ ]:


current_batch


# In[ ]:


np.append(current_batch[:,1:,:],[[[99]]],axis=1)


# **NOTE: PAY CLOSE ATTENTION HERE TO WHAT IS BEING OUTPUTED AND IN WHAT DIMENSIONS. ADD YOUR OWN PRINT() STATEMENTS TO SEE WHAT IS TRULY GOING ON!!**

# In[ ]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[ ]:


test_predictions


# In[ ]:


scaled_test


# ## Inverse Transformations and Compare

# In[ ]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[ ]:


true_predictions


# In[ ]:


test


# In[ ]:


# IGNORE WARNINGS
test['Predictions'] = true_predictions


# In[ ]:


test


# In[ ]:


test.plot(figsize=(12,8))


# ## Adding in Early Stopping and Validation Generator

# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# In[ ]:


length = 49
generator = TimeseriesGenerator(scaled_train,scaled_train,
                               length=length,batch_size=1)


validation_generator = TimeseriesGenerator(scaled_test,scaled_test,
                                          length=length,batch_size=1)


# # LSTMS

# In[ ]:


# define model
model = Sequential()

# Simple RNN layer
model.add(LSTM(50,input_shape=(length, n_features)))

# Final Prediction
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# In[ ]:


model.fit_generator(generator,epochs=20,
                   validation_data=validation_generator,
                   callbacks=[early_stop])


# In[ ]:


test_predictions = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[ ]:


# IGNORE WARNINGS
true_predictions = scaler.inverse_transform(test_predictions)
test['LSTM Predictions'] = true_predictions
test.plot(figsize=(12,8))


# # Forecasting
# 
# Forecast into unknown range. We should first utilize all our data, since we are now forecasting!

# In[ ]:


full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)


# In[ ]:


length = 50 # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)


# In[ ]:


model = Sequential()
model.add(LSTM(50, input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator,epochs=6) #6 because it stopped early last time at 6


# In[ ]:


forecast = []

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):
    
    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    forecast.append(current_pred) 
    
    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[ ]:


forecast = scaler.inverse_transform(forecast)


# In[ ]:


# forecast


# In[ ]:


df


# In[ ]:


len(forecast)


# In[ ]:


50*0.1


# In[ ]:


forecast_index = np.arange(50.1,55.1,step=0.1)


# In[ ]:


len(forecast_index)


# In[ ]:


plt.plot(df.index,df['Sine'])
plt.plot(forecast_index,forecast)


# # Please Upvote my kernel if you like it.
