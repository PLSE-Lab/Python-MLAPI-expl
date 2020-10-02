#!/usr/bin/env python
# coding: utf-8

# **Linear Regressor using Tensorflow**
# 
# This notebook is to have an understanding of how to implement a simple linear regressor (single feature) in tensorflow.
# X and Y are calculated using linspace and a slight noise will be introduced. In the last cell, tou can see the derived linear curve solution.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[ ]:


#create X and Y
x = np.linspace(0,50,50) + np.random.uniform(-3,3,50)
y = np.linspace(0,50,50)+ np.random.uniform(-3,5,50)

plt.scatter(x,y)
plt.title('training')
plt.show()


# In[ ]:


#X & Y are placeholders , to hold training data on runtime. We'll use stochastic gradient descend (add one training data at a time)

X = tf.placeholder("float")
Y = tf.placeholder("float")

#W & b are variables , which have an initialized value and will be changed during training
W = tf.Variable(np.random.rand())
b = tf.Variable(np.random.rand())


# In[ ]:


epochs = 1000
learning_rate = .01


# In[ ]:


y_pred = tf.add(tf.multiply(W,X),b)
cost = tf.reduce_sum(tf.pow(y_pred - Y,2))/100
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[ ]:


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for _x,_y in zip(x,y):
            #optimizer will be called , that causes the weights to get corrected
            sess.run(optimizer,feed_dict={X:_x,Y:_y})
        if(epoch%50 == 0):
            print('cost :',sess.run(cost,feed_dict={X:x,Y:y}))
    finalW = sess.run(W)
    finalb = sess.run(b)
predictions = finalW*x + finalb


# In[ ]:


#Orange points in the below plots show the prediction
plt.scatter(x,y)
plt.scatter(x,predictions)
plt.show()


# In[ ]:




