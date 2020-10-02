#!/usr/bin/env python
# coding: utf-8

# # Time series data for predicting monthly  milk production

# Monthly milk production: pounds per cow. Jan 62 - Dec 75

# <img src="http://slideplayer.com/slide/9823679/32/images/1/The+Process+of+Milk+Production.jpg" />

# ### importing libraries and time series data

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


milk = pd.read_csv('/kaggle/input/monthy-milk/monthly-milk-production.csv',index_col='Month')


# In[ ]:


milk.head()


# In[ ]:


milk.index = pd.to_datetime(milk.index)


# In[ ]:


milk.describe()


# In[ ]:


milk.plot()


# In[ ]:


milk.info()


# In[ ]:


train_set = milk.head(156)


# In[ ]:


train_set.head()


# In[ ]:


test_set = milk.tail(12)


# ## Scale the Data

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


train_scaled = scaler.fit_transform(train_set)


# In[ ]:


test_scaled = scaler.transform(test_set)


# ## Batch Function

# In[ ]:


def next_batch(training_data,batch_size,steps):
    
    
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0,len(training_data)-steps) 

    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)

    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


# In[ ]:


# y_batch[:<all rows>, :-1<from last>].reshape(-1, steps, 1), y_batch[:, 1:<from start>].reshape(-1, steps, 1) 
# we need 2 y_batch because of back propogating


# ## MODEL LSTM

# <img src='https://isaacchanghau.github.io/img/deeplearning/lstmgru/lstmandgru.png' />

# In[ ]:


# putting values to parameter
num_inputs = 1
num_time_steps = 12
num_neurons = 100
num_outputs = 1
learning_rate = 0.03
num_train_iterations = 4000
batch_size = 1


# In[ ]:


X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])


# ** Now create the RNN Layer, you have complete freedom over this, use tf.contrib.rnn and choose anything you want, OutputProjectionWrappers, BasicRNNCells, BasicLSTMCells, MultiRNNCell, GRUCell etc... Keep in mind not every combination will work well! (If in doubt, the solutions used an Outputprojection Wrapper around a basic LSTM cell with relu activation.**

# In[ ]:


cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs) 


# In[ ]:


# tf.nn.rnn_cell.LSTMCell above


# In[ ]:


outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


# In[ ]:


# above is deprecated use this one ===tf.keras.layers.RNN


# In[ ]:


loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)


# In[ ]:


init = tf.global_variables_initializer()


# In[ ]:


saver = tf.train.Saver()


# In[ ]:


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)


# In[ ]:


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        
        X_batch, y_batch = next_batch(train_scaled,batch_size,num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})
        
        if iteration % 100 == 0:
            
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    # Save Model for Later
    saver.save(sess, "./ex_time_series_model")


# # Generative Session
# 
# NOTE: generating new values based off some previous pattern, rather than trying to directly predict the future.  its limits due to the smaller size of our data set.

# In[ ]:


with tf.Session() as sess:
    
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./ex_time_series_model")

    # Create a numpy array for your genreative seed from the last 12 months of the 
    # training set data. Hint: Just use tail(12) and then pass it to an np.array
    train_seed = list(train_scaled[-12:])
    
    ## Now create a for loop that 
    for iteration in range(12):
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        train_seed.append(y_pred[0, -1, 0])


# In[ ]:


# inverse transform data into normal form from scaling


# In[ ]:


results = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))


# In[ ]:


#new column on the test_set = "Generated" 


# In[ ]:


test_set['Generated'] = results


# In[ ]:


test_set.plot()


# In[ ]:


test_set

