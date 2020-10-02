#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


stock_df = pd.read_csv('../input/prices-split-adjusted.csv',index_col='date')


# In[3]:


stock_df.head()


# In[4]:


stock_df = stock_df[stock_df['symbol'] == 'AMZN']
stock_df.drop(['symbol','open','low','high','volume'],inplace=True,axis=1)


# In[5]:


stock_df.info()


# In[7]:


stock_df.plot(figsize=(12,8),legend=True)


# In[8]:


def index_year(date_year):
  count = 0
  for dates in date_year:
    date_y = dates.split('-')   
    if date_y[0] == '2016':
        count += 1
        
  print("Data that is collected for the year 2016 is:::",count)


# In[9]:


try:
 stock_df.apply(index_year(stock_df.index))
except:
       print("")


# In[10]:


#### Train , test split

train_df = stock_df[:-252]
test_df = stock_df[-252:]


# In[11]:


### Scaling the data.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_scaled_df = scaler.fit_transform(train_df)
test_scaled_df = scaler.transform(test_df)


# In[12]:


### Creating a batch function

def next_batch(training_data,steps,batch_size):
    
    rand_start = np.random.randint(len(training_data) - steps)
    
    y_batch = np.array(training_data[rand_start:rand_start + steps + 1]).reshape(1,steps+1)
    #print(y_batch)
    return y_batch[:,:-1].reshape(-1,steps,1),y_batch[:,1:].reshape(-1,steps,1)


# In[13]:


num_inputs = 1
num_outputs = 1
num_neurons = 2000
batch_size = 1
num_iterations = 200
num_time_steps = 20
learning_rate = 0.001


# In[14]:


X = tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])


# In[15]:


#basic_cell = tf.contrib.rnn.OutputProjectionWrapper(
 #   tf.contrib.rnn.BasicRNNCell(num_units=num_neurons,activation=tf.nn.relu),output_size=num_outputs)


# In[16]:


#basic_outputs,basic_states = tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32)


# In[ ]:


lstm_cell = tf.contrib.rnn.OutputProjectionWrapper(
             tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons,activation=tf.nn.relu),output_size=num_outputs)


# In[ ]:


lstm_outputs, lstm_states = tf.nn.dynamic_rnn(lstm_cell,X,dtype=tf.float32)


# In[17]:


loss = tf.reduce_mean(tf.square(lstm_outputs - y))
#tf.reduce_mean(tf.square(basic_outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_err = optimizer.minimize(loss)


# In[18]:


init = tf.global_variables_initializer()


# In[19]:


saver = tf.train.Saver()


# In[ ]:


with tf.Session() as sess:
    
    sess.run(init)
    
    for iterations in range(num_iterations):
        
        x_batch, y_batch = next_batch(train_scaled_df,num_time_steps,batch_size)
        
        sess.run(train_err,feed_dict={X:x_batch,y:y_batch})
        
        if iterations%10 == 0:
            
            mse = loss.eval(feed_dict={X:x_batch,y:y_batch})
            
            print(iterations,'MSE is:',mse)
    
          
    saver.save(sess,'./stock_model')


# In[21]:


### Predictions

with tf.Session() as sess:
    
    saver.restore(sess,"./stock_model")
    
    train_seed = list(test_scaled_df[:252])
    for iterations in range(len(test_scaled_df)):
        
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1,num_time_steps,1)
        y_pred = sess.run(lstm_outputs,feed_dict={X:X_batch})
        #(basic_outputs,feed_dict={X:X_batch})
        #
        #
        #print(y_pred)
        train_seed.append(y_pred[0,-1,0])


# In[22]:


results = scaler.inverse_transform(np.array(train_seed[252:]).reshape(252,1))


# In[23]:


test_df['Predictions'] = results


# In[24]:


test_df.head()


# In[ ]:





# In[ ]:




