#!/usr/bin/env python
# coding: utf-8

# In[110]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sbs

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[111]:


trainDf = pd.read_csv('../input/train.csv').dropna()
testDf = pd.read_csv('../input/test.csv')


# In[113]:


trainDf.plot.scatter('x', 'y', s=1)


# In[114]:


sess = tf.Session()


# In[115]:


x = tf.placeholder( tf.float64, shape=(1, None), name='x')
y = tf.placeholder( tf.float64, shape=(1, None), name='y')
w = tf.Variable(tf.random_normal((1,1), dtype=tf.float64))
b = tf.Variable(tf.random_normal((1,1), dtype=tf.float64))
y_hat = tf.matmul(w, x) + b
loss = tf.reduce_sum(tf.pow(y_hat - y, 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00000002)
optimizer_node = optimizer.minimize(loss)
initializer = tf.global_variables_initializer()


# In[116]:


sess.run(initializer)
w_val, b_val, loss_val = sess.run([w, b, loss], feed_dict = {x: trainDf['x'].values.reshape(1, -1), y: trainDf['y'].values.reshape(1, -1)})
print(w_val, b_val, loss_val)
for _ in range(100):
    sess.run([optimizer_node], feed_dict = {x: trainDf['x'].values.reshape(1, -1), y: trainDf['y'].values.reshape(1, -1)})
    w_val, b_val, loss_val = sess.run([w, b, loss], feed_dict = {x: trainDf['x'].values.reshape(1, -1), y: trainDf['y'].values.reshape(1, -1)})
    print(w_val, b_val, loss_val)


# In[117]:


w_val, b_val = sess.run([w, b])
x_sweep = np.linspace(0, 100, 20)
y_sweep = w_val[0] * x_sweep + b_val[0]
plt.scatter(trainDf['x'], trainDf['y'], s=1)
plt.plot(x_sweep, y_sweep, 'r')


# In[118]:


w_val, b_val, loss_val = sess.run([w, b, loss], feed_dict = {x: testDf['x'].values.reshape(1, -1), y: testDf['y'].values.reshape(1, -1)})
print(w_val, b_val, loss_val)

