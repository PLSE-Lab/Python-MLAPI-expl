#!/usr/bin/env python
# coding: utf-8

# # Accelerating Scikit-Learn's Linear Regression using Tensorflow
# 
# In this short kernel, we will compare the computation speed of Scikit-Learn's linear regression (more precisely, Ridge regression) with a custom Tensorflow graph. The latter will be using GPUs to learn the weights, which decreases training time if you use the right solver.
# 
# [This gist](https://gist.github.com/xhlulu/334c24933e6f4913f4d779b784e71043) gives you an object-oriented version of this kernel, with an API similar to Scikit-Learn.

# In[ ]:


import os
import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import tensorflow as tf


# # Generating Dataset
# 
# We will generate a large synthetic dataset, then split it into two sets. We add a bias in order to test whether `fit_intercept` works for our Tensorflow model.

# In[ ]:


X, y = make_regression(
    n_samples=100000, 
    n_features=5000,
    n_informative=1000,
    random_state=2019,
    bias=5,
    noise=10
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2019
)


# # Using Scikit-Learn
# 
# Below, we show the training time of `sklearn.linear_model.Ridge` on a single CPU core. `Ridge` performs the same task as `LinearRegression`, except you can additionally use L2 regularization, and use a different underlying solver.

# In[ ]:


model = Ridge(solver='cholesky', fit_intercept=True, alpha=0)
get_ipython().run_line_magic('time', 'model.fit(X_train, y_train)')


# We print out the MSE as well as a sanity check.

# In[ ]:


get_ipython().run_line_magic('time', 'y_pred_sklearn = model.predict(X_test)')
print(mean_squared_error(y_test, y_pred_sklearn))


# # Building a graph with Tensorflow
# 
# Now, we use build a tensorflow graph, and use the built-in `tf.linalg.lstsq`. Here, setting `fast=True` means that we use Cholesky decomposition, which speeds up subsequent computations (matrix multiplication and inversion).

# In[ ]:


fit_intercept = True

graph = tf.Graph()
with graph.as_default():
    tf_y = tf.placeholder(tf.float64, shape=(None, None))
    tf_input = tf.placeholder(tf.float64, shape=(None, None), name='input')
    
    if fit_intercept:
        tf_bias = tf.ones((tf.shape(tf_input)[0], 1), dtype=tf.float64)
        tf_x = tf.concat([tf_input, tf_bias], axis=1)
    else:
        tf_x = tf_input
    
    tf_weights = tf.linalg.lstsq(tf_x, tf_y, fast=True)
    
    tf_trained_weights = tf.placeholder(tf.float64, shape=(None, None))
    tf_preds = tf.matmul(tf_x, tf_trained_weights)

# training
with tf.Session(graph=graph) as sess:
    get_ipython().run_line_magic('time', 'weights = sess.run(tf_weights, feed_dict={tf_input: X_train, tf_y: np.expand_dims(y_train, axis=-1)})')
    
# Predicting
with tf.Session(graph=graph) as sess:
    get_ipython().run_line_magic('time', 'y_pred_tf = sess.run(tf_preds, feed_dict={tf_input: X_test, tf_trained_weights: weights})')

mean_squared_error(y_test, y_pred_tf)


# We notice that our Tensorflow implementation is significantly faster than scikit-learn, and produces the same result. This is likely due to how Tensorflow performs Cholesky decomposition, which is a well-known method used in Computer Animation and [optimized for GPUs](https://www.dusers.drexel.edu/~julian/experience/classwork/gpu-cholesky-decomposition/).
# 
# You can go ahead and try to set `fast=False`. You will realize that it is much slower, which probably indicates that Tensorflow does not handle pure matrix inversion as well as `scipy`.
