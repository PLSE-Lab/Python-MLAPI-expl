#!/usr/bin/env python
# coding: utf-8

# NOTE: I rewrite various notebooks because that's how I learn. I do it on Kaggle because I like their community and other features. Please use and credit original source.
# 
# Source: https://github.com/tensorflow/docs/blob/master/site/en/r2/guide/autograph.ipynb

# # tf.function and AutoGraph in TensorFlow 2.0

# TF 2.0 bring together the ease of eager execution and the power of TF 1.0. At the center of this merger is `tf.function` which allows you to transform a subset of Python syntax into protable, high-performance TensorFlow graphs.
# 
# A cool new feature of `tf.function` is AutoGraph, which lets you write graph code using natural Python syntax. For a list of the Python features that you can use with AutoGraph, see [AutoGraph capabilities and limitations](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/LIMITATIONS.md). For more details aabout `tf.function` see the RFC [TF 2.0: Functions, not Sessions](https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md). For more details about Autograph, see [tf.autograph](https://www.tensorflow.org/api_docs/python/tf/autograph).
# 
# This tutorial will walk you through the basic features of `tf.function` and AutoGraph.

# ## Setup
# 
# Import TensorFlow and enable TF 2.0 mode:

# In[ ]:


import sys
print("Python version:", sys.version)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# In[ ]:


from __future__ import absolute_import, division, print_function
import numpy as np

import tensorflow as tf


# Install aa temporary patch to enable a few extra TF 2.0 upgrades. This piece will be removed soon.

# In[ ]:


from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


# ## The `tf.function` decorator
# 
# When you annotate a function with `tf.function`, you can still call it like any other function. But it will be compiled into a graph, which means you get the benefits of faster execution, running on GPU or TPU, or exporting to SavedModel.

# In[ ]:


@tf.function
def simple_nn_layer(x, y):
    return tf.nn.relu(tf.matmul(x, y))

x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

simple_nn_layer(x, y)


# If we examine the result of the annotation, we can see that it's a special callable that handles all interactions with the TensorFlow runtime.

# In[ ]:


simple_nn_layer


# If your code uses multiple functions, you don't need to annotate them all - any functions called from an annotated function will also run in graph mode.

# In[ ]:


def linear_layer(x):
    return 2 * x + 1

@tf.function
def deep_net(x):
    return tf.nn.relu(linear_layer(x))

deep_net(tf.constant((1, 2, 3)))


# ## Use Python control flow
# 
# When using data-dependent control flow inside `tf.function`, you can use Python control flow statements and AutoGraph will convert them into appropriate TensorFlow ops. For example, `if` statements will be converted into `tf.cond()` if they depend on a `Tensor`.
# 
# In the example below, x is a `Tensor` but the `if` statement works as expected:

# In[ ]:


@tf.function
def square_if_positive(x):
    if x > 0:
        x = x * x
    else:
        x = 0
    return x

print('square_if_positive(2) = {}'.format(square_if_positive(tf.constant(2))))
print('square_if_positive(-2) = {}'.format(square_if_positive(tf.constant(-2))))


# Note: the example above shows how to perform simple conditionals when scalar values are involved. Typical ML code involves batches; in those cases you should consider using the faster and vecotrized `tf.where` if possible.
# 
# AutoGraph supports common Python statements like `while`, `for`, `if`, `break`, `continue`, `return`, with support for nesting. That means you  can use `Tensor` expressions in the condition of `while` and `if` statements, or iterate over a `Tensor` in a `for` loop.

# In[ ]:


@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s

sum_even(tf.constant([10, 12, 15, 20]))


# AutoGraph also provides a low-level API for advanced users. For example, we can use it to have a look at the generated code.

# In[ ]:


print(tf.autograph.to_code(sum_even.python_function, experimental_optional_features=None))


# Here's an example of more complicated control flow:

# In[ ]:


@tf.function
def fizzbuzz(n):
    msg = tf.constant('')
    for i in range(n):
        if i % 3 == 0:
            msg += 'Fizz'
        elif i % 5 == 0:
            msg += 'Buzz'
        else:
            msg += tf.as_string(i)
        msg += '\n'
    return msg

print(fizzbuzz(tf.constant(15)).numpy().decode())


# ## Use Python print
# 
# AutoGraph will also convert Python builtins like `print`.
# 
# Note: due to the parallel nature of calculations in TensorFlow, statements might execute out of order. It's best to use `print` only to inspect actual values, and you should not use it to determine whether the program execution reaches a certain point. (What to use for that?)

# In[ ]:


@tf.function
def count(n):
    for i in tf.range(n):
        print(i)
        
count(tf.constant(5))


# ## Other handy conversions
# 
# Other builtins that AutoGraph can adapt for TensorFlow are `range` and `len`.
# 
# `range` is a shortcut for `tf.range`:

# In[ ]:


@tf.function
def range_example(n):
    return range(n)

print(range_example(tf.constant(3)))


# `len` is a shortcut for `.shape[0]`:

# In[ ]:


@tf.function
def len_example(n):
    return len(n)

print(len_example(tf.zeros((20, 10))))


# ## Kears and AutoGraph
# 
# You can use `tf.function` with object methods as well. For example, you can decorate your custom Keras models, typically by annotating the model's `call` function. For more information, see `tf.keras`.

# In[ ]:


class CustomModel(tf.keras.models.Model):
    
    @tf.function
    def call(self, input_data):
        if tf.reduce_mean(input_data) > 0:
            return input_data
        else:
            return input_data // 2
        
model = CustomModel()
model(tf.constant([-2, -4]))


# ## Side effects
# 
# Just like in eager mode, you can use operations with side effects, like `tf.assign` or `tf.print` normally inside `tf.function`, and it will insert the necessary control dependencies to ensure they execute in order.

# In[ ]:


v = tf.Variable(5)

@tf.function
def find_next_odd():
    v.assign(v + 1)
    if v % 2 == 0:
        v.assign(v + 1)

find_next_odd()
v


# ## Example: training a simple model
# 
# AutoGraph also allows you to mode more compuration inside TensorFlow, For example, a training loop is just control flow, so it can actually be brought into TensorFlow.

# ### Download data

# In[ ]:


def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    
    return x, y

def mnist_dataset():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds

train_dataset = mnist_dataset()


# ### Define the model

# In[ ]:


model = tf.keras.Sequential((
    tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10)))

model.build()
optimizer = tf.keras.optimizers.Adam()


# ### Define the training loop

# In[ ]:


def compute_loss(logits, labels):
    return tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(predictions == labels, tf.float32))

def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        tape.watch(model.variables)
        logits =  model(x)
        loss = compute_loss(logits, y)
        
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))
    
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy


@tf.function
def train(model, optimizer):
    train_ds = mnist_dataset()
    step = 0
    for x, y in train_ds:
        step += 1
        loss, accuracy = train_one_step(model, optimizer, x, y)
        if step % 10 == 0:
            print('Step', step, ': loss', loss, ':, accuracy', accuracy)
    return step

_ =  train(model, optimizer)


# ## A note on batching
# 
# In real applications batching is essential for performance. The best code to convert to AutoGraph is code where the control flow is decided at the *batch* level. If making decisions at the individual *example* level, try to use batch APIs to maintain performance.
# 
# For example, if you have the following code in Python:

# In[ ]:


def square_if_positive(x):
    return [i ** 2 if i > 0 else i for i in x]

square_if_positive(range(-5, 5))


# You may be tempted to write it in TensorFlow as such (and this would work!):

# In[ ]:


@tf.function
def square_if_positive_naive(x):
    result = tf.TensorArray(tf.int32, size=len(x))
    for i in range(len(x)):
        if x[i] > 0:
            result = result.write(i, x[i] ** 2)
        else:
            result = result.write(i, x[i])
    return result.stack()

square_if_positive_naive(tf.range(-5, 5))


# But in this case, it turns out you can write the following:

# In[ ]:


def square_if_positive_vectorized(x):
    return tf.where(x > 0, x ** 22, x)

square_if_positive_vectorized(tf.range(-5, 5))

