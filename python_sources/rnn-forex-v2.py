#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data_history = [] #do not refresh this unless you want to delete the history pf parameters

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import math


# In[ ]:


# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


# In[ ]:


df = pd.read_csv("../input/1_MIN_ALL.txt", index_col=0, sep = ' ')
df = df.tail(1088480)
df = df.drop('Per',axis = 1)
df['Vol'] = df['Vol'].str.replace("'", '')
df['DateTime'] = df['Date']*1000000 + df['Time']
df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y%m%d%H%M%S")
df = df.reset_index(drop = True)
df = df.set_index('DateTime')
df = df.drop(['Date','Time'],axis = 1)
df = df
df = df.head(300000)
df.head()


# ### Adding data to help the model deal with seasonality

# In[ ]:


#df['DayOfYear'] = df.index.dayofyear
df['HourOfDay'] = df.index.hour
#df['MonthOfYear'] = df.index.month


# In[ ]:


df = df.fillna(method='ffill')


# In[ ]:


df[['Open','High','Low','Close']] = np.exp(10*df[['Open','High','Low','Close']]) #See if we can help the model out


# In[ ]:


df.describe()


# In[ ]:


df.values.shape


# Selecting the columns to predict

# In[ ]:


target_names = ['Close']


# Shifting the data 

# In[ ]:


shift_mn = 1
shift_steps = shift_mn * 1  # Number of mn.


# In[ ]:


df_targets = df[target_names].shift(-shift_steps)


# ### NumPy Arrays
# Since we shifted the data 1 mn we have to delete the last 1 rows

# In[ ]:


x_data = df.values[0:-shift_steps]
x_data = x_data.astype('float32')


# In[ ]:


print(type(x_data))
print("Shape:", x_data.shape)


# In[ ]:


y_data = df_targets.values[:-shift_steps]
y_data = y_data.astype('float32')


# In[ ]:


print(type(y_data))
print("Shape:", y_data.shape)


# In[ ]:


#nb of data rows in the dataset
num_data = len(x_data)
num_data


# In[ ]:


train_split = 0.9


# In[ ]:


#This is the number of observations in the test-set
num_train = int(train_split * num_data)
num_train


# In[ ]:


#These are the input-signals for the training- and test-sets
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)


# In[ ]:


#These are the output-signals for the training- and test-sets
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
len(y_train) + len(y_test)


# In[ ]:


#number of input-signals
num_x_signals = x_data.shape[1]
num_x_signals


# In[ ]:


#number of output-signals
num_y_signals = y_data.shape[1]
num_y_signals


# # Scaling Data
# We scale the data to a range between 0 and 1 to help the model perform better.

# In[ ]:


print("Min:", np.min(x_train))
print("Max:", np.max(x_train))


# In[ ]:


x_scaler = MinMaxScaler()


# In[ ]:


x_train_scaled = x_scaler.fit_transform(x_train)


# In[ ]:


print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))


# In[ ]:


x_test_scaled = x_scaler.transform(x_test)


# In[ ]:


y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# # Generating Data
# The data-set has now been prepared as 2-dimensional numpy arrays.

# In[ ]:


print(x_train_scaled.shape)
print(y_train_scaled.shape)


# 
# Instead of training the Recurrent Neural Network on the complete sequences of almost 300k observations, we will use the following function to create a batch of shorter sub-sequences picked at random from the training-data.

# In[ ]:


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)


# We adabt the batch size to use a maximum of the CPU (near 100%)

# In[ ]:


batch_size = 64


# In[ ]:


# We adjust the length to be  interesting for the model to work on it but not to heavy to crush our computer
sequence_length = 60 * 24 * 7
sequence_length


# In[ ]:


generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)


# We test the batch generator to see if it works

# In[ ]:


x_batch, y_batch = next(generator)


# In[ ]:


print(x_batch.shape)
print(y_batch.shape)


# In[ ]:


#Lets plot it 

batch = 0   # First sequence in the batch.
signal = 0  # First signal from the input-signals.
seq = x_batch[batch, :, signal]
plt.plot(seq)


# In[ ]:


#plot of the batch we want to predict
seq = y_batch[batch, :, signal]
plt.plot(seq)


# ## Validation Set

# In[ ]:


validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


# # Create the RNN

# In[ ]:


model = Sequential()


# In[ ]:


model.add(GRU(units= batch_size*2,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))


# In[ ]:


# The GRU outputs a batch of sequences of 512 values. We want to predict 1 output-signals, 
# so we add a fully-connected (or dense) layer which maps 512 values down to only 1 values.
model.add(Dense(num_y_signals, activation='sigmoid'))


# A problem with using the Sigmoid activation function, is that we can now only output values in the same range as the training-data.
# For example, if the training-data only has values between 0.8 and 1.3 then the scaler will map 0.8 to 0 and 1.3 to 1. So if we limit the output of the neural network to be between 0 and 1 using the Sigmoid function, this can only be mapped back to temperature values between 0.8 and 1.3.
# 
# We can use a linear activation function on the output instead. This allows for the output to take on arbitrary values. It might work with the standard initialization for a simple network architecture, but for more complicated network architectures e.g. with more layers, it might be necessary to initialize the weights with smaller values to avoid NaN values during training. You may need to experiment with this to get it working.

# In[ ]:


if False:
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))


# ## Loss Function
# We will use Mean Squared Error (MSE) as the loss-function that will be minimized. This measures how closely the model's output matches the true output signals.
# However, at the beginning of a sequence, the model has only seen input-signals for a few time-steps, so its generated output may be very inaccurate. Using the loss-value for the early time-steps may cause the model to distort its later output. We therefore give the model a "warmup-period" of 50 time-steps where we don't use its accuracy in the loss-function, in hope of improving the accuracy for later time-steps.
# 

# In[ ]:


warmup_steps = 50


# In[ ]:


def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


# # Compiling the model
# This is a very small model with only two layers. The output shape of (None, None, 3) means that the model will output a batch with an arbitrary number of sequences, each of which has an arbitrary number of observations, and each observation has 3 signals. This corresponds to the 3 target signals we want to predict.
# 
# 

# In[ ]:


optimizer = RMSprop(lr=1e-3)


# In[ ]:


model.compile(loss=loss_mse_warmup, optimizer=optimizer)


# In[ ]:


model.summary()


# ## Callbacks 
# During training we want to save checkpoints and log the progress to TensorBoard so we create the appropriate callbacks for Keras.
# This is the callback for writing checkpoints during training.

# In[ ]:


#This is the callback for writing checkpoints during training.
path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)


# In[ ]:


#stopping the model when performance worsens on the valid set
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)


# In[ ]:


#This is the callback for writing the TensorBoard log during training.
callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)


# In[ ]:


#This callback reduces the learning-rate for the optimizer if the validation-loss has not improved since 
# the last epoch (as indicated by patience=0). The learning-rate will be reduced by multiplying it with 
# the given factor. We set a start learning-rate of 1e-3 above, so multiplying it by 0.1 gives a learning-rate of 1e-4. 
# We don't want the learning-rate to go any lower than this.

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)


# In[ ]:


callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


# # Train the Recurrent Neural Network
# Don't forget to activate the GPU if you're using the kaggle kernel ! It saves hell of a time !!!

# In[ ]:


epochs = 20
steps_per_epoch = 100


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(generator=generator,\n                    epochs=epochs,\n                    steps_per_epoch=steps_per_epoch, #should be 100 ##################\n                    validation_data=validation_data,\n                    callbacks=callbacks)')


# ## Load Checkpoint
# Because we use early-stopping when training the model, it is possible that the model's performance has worsened on the test-set for several epochs before training was stopped. We therefore reload the last saved checkpoint, which should have the best performance on the test-set.
# 

# In[ ]:


try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)


# ## Performance on Test-Set
# We can now evaluate the model's performance on the test-set. This function expects a batch of data, but we will just use one long time-series for the test-set, so we just expand the array-dimensionality to create a batch with that one sequence.
# 

# In[ ]:


loss_test_set = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))


# In[ ]:


print("loss (test-set):", loss_test_set)


# In[ ]:


# If you have several metrics you can use this instead.
if False:
    for res, metric in zip(result, model.metrics_names):
        print("{0}: {1:.3e}".format(metric, res))


# ## Generate prediction

# In[ ]:


def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15,5))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        
        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()


# ### Example from Test-Set
# 
# Now consider an example from the test-set. The model has not seen this data during training.
# The temperature is predicted reasonably well, although the peaks are sometimes inaccurate.
# The wind-speed has not been predicted so well. The daily oscillation-frequency seems to match, but the center-level and the peaks are quite inaccurate. A guess would be that the wind-speed is difficult to predict from the given input data, so the model has merely learnt to output sinusoidal oscillations in the daily frequency and approximately at the right center-level.
# The atmospheric pressure is predicted reasonably well, except for a lag and a more noisy signal than the true time-series.

# In[ ]:


plot_comparison(start_idx=200000, length=10000, train=True)


# In[ ]:


plot_comparison(start_idx=200000, length=1000, train=True)


# In[ ]:


def return_pred(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    y_pred_rescaled[:,0] = np.log(y_pred_rescaled[:,0])/10
    y_tc = y_true[:,0]
    y_tc = np.log(y_true[:,0])/10
    
    result = pd.DataFrame({'Close_pred':y_pred_rescaled[:,0],'Close_true':y_tc})
    result['rmse'] = np.sqrt((result['Close_pred'] - result['Close_true'])*
                              (result['Close_pred'] - result['Close_true']))
    
    result = result.tail(len(result)-200) #giving at least 50 learning steps to the model
    return(result)


# In[ ]:


result = return_pred(start_idx = 0, length=10000, train=False)
result.head()


# In[ ]:


result.describe()


# In[ ]:


rmse = result.describe()['rmse'][1]
data_history.append([rmse,loss_test_set,shift_mn,epochs,steps_per_epoch,batch_size,sequence_length])
df = pd.DataFrame(data_history,columns=['rmse','trainScore','shift_mn',
                                        'epochs','steps_per_epoch','batch_size','sequence_length'])
df


# # License (MIT)
# Copyright (c) 2018 by Magnus Erik Hvass Pedersen
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# In[ ]:





# In[ ]:




