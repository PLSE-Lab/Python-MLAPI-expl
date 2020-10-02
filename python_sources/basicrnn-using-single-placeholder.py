#!/usr/bin/env python
# coding: utf-8

# In[19]:


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


# In[20]:


import tensorflow as tf


# In[21]:


tf.reset_default_graph()


# In[22]:


n_inputs = 3
n_neurons = 5
n_steps = 2


# In[23]:


X = tf.placeholder(tf.float32,[None,n_steps,n_inputs]) #mini-batch size,time steps,input sequence


# In[24]:


X_seqs = tf.unstack(tf.transpose(X,perm=[1,0,2]))


# #### X_seqs is a Python list of n_steps tensors of shape [None, n_inputs], where once again the first dimension is the mini-batch size. To do this, we first swap the first two dimensions using the transpose() function, so that the time steps are now the first dimension. Then we extract a Python list of tensors along the first dimension (i.e., one tensor per time step) using the unstack() function

# In[25]:


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)


# In[26]:


output_seqs , states = tf.contrib.rnn.static_rnn(basic_cell , X_seqs , dtype=tf.float32)


# In[27]:


outputs = tf.transpose(tf.stack(output_seqs),perm=[1,0,2])


# In[28]:


X_batch = np.array([
    # t=0     t=1    
    [[0,1,2],[9,8,7]],     #instance 0
    [[3,4,5],[0,0,0]],     #instance 1
    [[6,7,8],[6,5,4]],     #instance 2
    [[9,0,1],[3,2,1]]      #instance 3
    
])


# In[29]:


init = tf.global_variables_initializer()


# In[30]:


with tf.Session() as sess:
    sess.run(init)
    outputs_val = outputs.eval(feed_dict={X:X_batch})
    


# In[31]:


print(outputs_val)


# In[ ]:





# In[ ]:




