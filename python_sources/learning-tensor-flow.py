#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


node1 =tf.constant(4,tf.float64)
node2 =tf.constant(6,tf.float64)
print(node1,node2)

sess =tf.Session()
print(sess.run([node1, node2]))


# In[ ]:


node3=tf.add(node1 , node2)
print("sess.run(node3): ",sess.run(node3))


# In[ ]:


a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder_node = a + b
print(sess.run(adder_node, {a: [5,9], b: [2, 4]}))
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))


# In[ ]:




W = tf.Variable([.1], tf.float32)
b = tf.Variable([.2], tf.float32)
x = tf.placeholder(tf.float32)

init = tf.global_variables_initializer()
sess.run(init)

linear_model = W * x + b
print(sess.run(linear_model, {x:[1,2,3,4,7,8]}))


# In[ ]:


features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)
estimator.fit(input_fn=input_fn, steps=1000)
print(estimator.evaluate(input_fn=input_fn))

