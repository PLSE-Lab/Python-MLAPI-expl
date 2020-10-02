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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


get_ipython().system('pip list')


# In[ ]:


get_ipython().system('cat /usr/local/cuda/version.txt')


# In[ ]:


get_ipython().system('cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2')


# In[ ]:


get_ipython().system('free -g')


# In[ ]:


get_ipython().system('cat /proc/cpuinfo')


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


get_ipython().system('ls ../input/mnist-cnn')


# In[ ]:


get_ipython().system('CUDA_VISIBLE_DEVICES=0 python "../input/mnist-cnn/mnist_cnn.py"')


# In[ ]:


get_ipython().system('python "../input/mnist-cnn256/mnist_cnn256.py"')


# In[ ]:


get_ipython().system('python "../input/multi-gpu/multigpu_cnn.py"')


# In[ ]:


import tensorflow as tf


# In[ ]:


# Creates a graph.
 
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
 
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
 
c = tf.matmul(a, b)
 
# Creates a session with log_device_placement set to True.
 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
 
# Runs the op.
 
print(sess.run(c)) 


# In[ ]:


# Creates a graph.
 
with tf.device('/cpu:0'):
 
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
 
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
 
  c = tf.matmul(a, b)
 
# Creates a session with log_device_placement set to True.
 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
 
# Runs the op.
 
print(sess.run(c))


# In[ ]:


# Creates a graph.
 
with tf.device('/gpu:0'):
 
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
 
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
 
  c = tf.matmul(a, b)
 
# Creates a session with log_device_placement set to True.
 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
 
# Runs the op.
 
print(sess.run(c))

