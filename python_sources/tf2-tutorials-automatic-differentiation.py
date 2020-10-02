#!/usr/bin/env python
# coding: utf-8

# NOTE: I rewrite various notebooks because that's how I learn. I do it on Kaggle because I like their community and other features. Please use and credit original source.
# Source: https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/eager/automatic_differentiation.ipynb

# # Automatic differentiation and gradient tape
# 
# In the previous tutorial we inroduced Tensors and operations on them. In this Tutorial we will cover [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), a key technique for optimizing machine learning models.

# In[ ]:


import sys
print("Python version:", sys.version)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# ## Gradient tapes
# 
# TensorFlow provides the [tf.GradientTape API](https://www.tensorflow.org/api_docs/python/tf/GradientTape) for automatic differentiation - computing the gradient of a computing with respect to its input variables. TensorFlow "records" all operations executed inside the context of a `tf.GradientTape` onto a "tape". TensorFlow then uses that tape and the gradients associated with each recorded operation to compute the gradients of a "recorded" computation using [reverse mode differeentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).
# 
# For example:

# In[ ]:


x = tf.ones((2, 2 ))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
    
# derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0


# You can also request gradients of the output with respect to intermediate values computed during a "recorded" `tf.GradientTape` context.

# In[ ]:


x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Use the tape to compute the derivative of z with respect to the intermedia value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0


# By default, the resources held by a `GradientTape` are released as soon as `GradientTape.gradient()` method is called. To compute multiple gradients over the same computation, create a `persistent` gradient tape. This allows multiple calls to the `gradient()` method, as resources are released when the tape object is garbage collected. For example:

# In[ ]:


x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y =  x * x
    z =  y * y

dz_dx = t.gradient(z, x) # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x) # 6.0
del t # Drop the reference to the tape


# ## Recording control flow
# 
# Because tapes record operations as they are executed, Python control flow (using `ifs` and `whiles` for example) is naturally handled:

# In[ ]:


def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0


# ## Higher-order gradients
# 
# Operations inside of the `GradientTape` context manageer are recorded for automatic differentiation. If gradients are computed in that context, then the gradient computation is recorded as well. As a result, the exact same API works for higher-order gradients as well. For example:

# In[ ]:


x = tf.Variable(1.0) # Create a TensorFlow variable initialized to 1.0
with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y =  x * x * x
    
    # compute the graident inside the 't' context manager which means the gradient computation is differentiable as well
    dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0


# **Next Steps**
# 
# In this tutorial we covered gradient computation in TensorFlow. WIth that we have enough of the primitives required to build and train neural networks.
