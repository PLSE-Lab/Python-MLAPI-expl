#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow==2.0.0-alpha0')

import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


# In[ ]:


print(tf.__version__)


# ## Eager Execution
# 
# Starting with version 2.0, Tensorflow supports eager execution by default. The main implication of this is that operations are evaluated immediately and return concrete values - without building any graphs.

# In[ ]:


print(tf.executing_eagerly())


# Confirmed! We are executing eagerly. Let's execute something:

# In[ ]:


x = [[2.]]
tf.matmul(x, x)


# Executing the Tensorflow op above gives us a tensor which references a concrete value (a numpy array), rether than a symbolic handle to a node in an execution graph. Cool.
# 
# Here are a few more examples:

# In[ ]:


print(tf.constant([[2.0],[1.0]]))


# In[ ]:


x = tf.constant(1.0)
x *= 10
print(x)
print(x.numpy())


# In[ ]:


print(tf.add(x, [[1.0],[2.0]]))


# Now, variables (i.e. `tf.Variable` objects) are mutable tensors which persist in memory during eager execution until all references to them are cleared.

# In[ ]:


v = tf.Variable(tf.random.normal([2, 1]))


# In[ ]:


v.assign([[1.0], [2.0]])


# Computation during eager execution is automatically sent to GPUs, but one can still control this using `tf.device` blocks. 
# 
# Eager execution is very handy for quick experimentation, but TensorFlow 1.X graph execution offered the possibility for distributed training, performance optimizations and production deployment of models. In TensorFlow 2.0, this gap is bridged with the use of the `tf.function` API and AutoGraph.

# ## tf.function and AutoGraph

# So what is this magic? Essentially, `@tf.function` can transfrom a subset of Python code into a TensorFlow graph, thus making it distributable, optimizable, saveable, deployable and all that good stuff. Let's see it in action:

# In[ ]:


@tf.function
def add_tensors(x, y):
    return tf.add(x, y)


# So we've defined a simple Python function, which works with tensors and which we can call in eager mode as we've been coding in this kernel so far. Proof:

# In[ ]:


# not even gonna bother with tf.constant. Much convenient, much eager
x = [1.0, 2.0]
y = [2.0, 3.0]
add_tensors(x, y)


# But we've also stuck a `tf.function` annotation to our function, which means the function will be compiled into a graph, which can be exported to a `SavedModel` or executed in distributed mode.
# 
# If your annotated functions call other functions, they don't all have to be annotated with `@tf.function`.

# You can use normal Python statements in the annotated functions and AutoGraph will convert them to TensorFlow ops. Let's try an `if` statement:

# In[ ]:


@tf.function
def simple_abs(x):
    if x < 0:
        -x
    else:
        x
print(tf.autograph.to_code(simple_abs.python_function, experimental_optional_features=None))


# Very cool! But beware, not all language features are supported. [Here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/LIMITATIONS.md) are the capabilities and limitations of AutoGraph.

# In[ ]:




