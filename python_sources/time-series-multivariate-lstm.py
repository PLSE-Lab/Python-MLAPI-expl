#!/usr/bin/env python
# coding: utf-8

# # Multivariate Time Series with RNN

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


# IGNORE THE CONTENT OF THIS CELL
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()


# ## Data
# 
# Let's read in the data set:

# In[ ]:


df = pd.read_csv('/kaggle/input/energydata_complete.csv',index_col='date',infer_datetime_format=True)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df['Windspeed'].plot(figsize=(12,8))


# In[ ]:


df['Appliances'].plot(figsize=(12,8))


# ## Train Test Split

# In[ ]:


len(df)


# In[ ]:


df.head(3)


# In[ ]:


df.tail(5)


# Let's imagine we want to predict just 24 hours into the future, we don't need 3 months of data for that, so let's save some training time and only select the last months data.

# In[ ]:


df.loc['2016-05-01':]


# In[ ]:


df = df.loc['2016-05-01':]


# Let's also round off the data, to one decimal point precision, otherwise this may cause issues with our network (we will also normalize the data anyways, so this level of precision isn't useful to us)

# In[ ]:


df = df.round(2)


# In[ ]:


len(df)


# In[ ]:


# How many rows per day? We know its every 10 min
24*60/10


# In[ ]:


test_days = 2


# In[ ]:


test_ind = test_days*144


# In[ ]:


test_ind


# In[ ]:


# Notice the minus sign in our indexing

train = df.iloc[:-test_ind]
test = df.iloc[-test_ind:]


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
length = 144 # Length of the output sequences (in number of timesteps)
batch_size = 1 #Number of timeseries samples in each batch
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)


# In[ ]:


len(scaled_train)


# In[ ]:


len(generator) 


# In[ ]:


# scaled_train


# In[ ]:


# What does the first batch look like?
X,y = generator[0]


# In[ ]:


print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# Now you will be able to edit the length so that it makes sense for your time series!

# ### Create the Model

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM


# In[ ]:


scaled_train.shape


# In[ ]:


# define model
model = Sequential()

# Simple RNN layer
model.add(LSTM(100,input_shape=(length,scaled_train.shape[1])))

# Final Prediction (one neuron per feature)
model.add(Dense(scaled_train.shape[1]))

model.compile(optimizer='adam', loss='mse')


# In[ ]:


model.summary()


# ## EarlyStopping

# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=1)
validation_generator = TimeseriesGenerator(scaled_test,scaled_test, 
                                           length=length, batch_size=batch_size)


# In[ ]:


model.fit_generator(generator,epochs=10,
                    validation_data=validation_generator,
                   callbacks=[early_stop])


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


first_eval_batch = first_eval_batch.reshape((1, length, scaled_train.shape[1]))


# In[ ]:


model.predict(first_eval_batch)


# In[ ]:


scaled_test[0]


# Now let's put this logic in a for loop to predict into the future for the entire test range.
# 
# ----

# **NOTE: PAY CLOSE ATTENTION HERE TO WHAT IS BEING OUTPUTED AND IN WHAT DIMENSIONS. ADD YOUR OWN PRINT() STATEMENTS TO SEE WHAT IS TRULY GOING ON!!**

# In[ ]:


n_features = scaled_train.shape[1]
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


true_predictions = pd.DataFrame(data=true_predictions,columns=test.columns)


# In[ ]:


true_predictions


# ### Lets save our model

# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


model.save("multivariate.h5")

