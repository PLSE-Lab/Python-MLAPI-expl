#!/usr/bin/env python
# coding: utf-8

# In[59]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[60]:


import tensorflow as tf


# In[61]:


tf.reset_default_graph()


# In[62]:


n_inputs = 3   # Assuming RNN runs over only two time steps taking input vector of size 3 at each time step
n_neurons = 5  # RNN composed of a layer of five recurrent neurons


# In[63]:


X0 = tf.placeholder(tf.float32,[None,n_inputs])
X1 = tf.placeholder(tf.float32,[None,n_inputs])


# In[64]:


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) 
# creates copies of the cell to build the unrolled RNN (one for each time step) with
# weights and bias terms and chains them


# In[65]:


output_seqs , states = tf.contrib.rnn.static_rnn(basic_cell , [X0,X1], dtype=tf.float32)
# output_seqs is a Python list containing the output tensors for each time step 
# states is a tensor containing the final states of the network
Y0 , Y1 = output_seqs


# In[66]:


# Mini-batch:        instance 0,instance 1,instance 2,instance 3
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1


# In[67]:


init = tf.global_variables_initializer()


# In[68]:


with tf.Session() as sess:
    sess.run(init)
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})


# In[69]:


print(Y0_val)


# In[70]:


print(Y1_val)


# In[ ]:




