#!/usr/bin/env python
# coding: utf-8

# NOTE: I rewrite various notebooks because that's how I learn. I do it on Kaggle because I like their community and other features. Please use and credit original source.
# 
# Source: https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/eager/custom_layers.ipynb

# # Custom Layers
# 
# We recommend using tf.keras as high-level API for building neural networks. That said, most TensorFlow APIs are usable with eager execution.

# In[ ]:


import sys
print("Python version:", sys.version)

import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# ## Layers: common set of useful operations
# 
# Most of the time when writing code for machine learning models you want to operate at a higher level of abstraction than individual operations and manipulations  of individual variables.
# 
# Many machine learning models are expressible ass the composition aand stacking of relatively simple layers, and TensorFlow provides both a set of many common layers as well ass easy ways for you to write your own application-specific layers either frrom scratch or as the composition of existing layers.
# 
# TensorFlow includes the full [Keras](https://keras.io/) API in `tf.keras` package, and the Keras layers are very useful when building your own models.

# In[ ]:


# In the tf.keras.layers package, layers are objects. To construct a layer, simply construct the object.
# Most layers take as aa first argument the number of output dimensions/channels.

layer = tf.keras.layers.Dense(100)

# the number of input dimensions is often unnecessary, as it can be inferred the first time the layer is used
# but it can be provided if you want to specify it manually, which is usefuly in some complex models.
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))


# The full list of pre-existing layers can be seen in [the documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers). It includes Dense (a fully-connected layer), Conv2D, LSTM, BatchNormalization, Dropout and many others.

# In[ ]:


# to use a layer, simply call it.
layer(tf.zeros([10, 5]))


# In[ ]:


# layers have many useful methods. For example, you can inspect all variables in a layer by calling layer.variables.
# in this case, a fully-connected layer will have variables for weights and biases
layer.variables


# In[ ]:


# the variables are also accessible through nice accessors
layer.kernel, layer.bias


# ## Implementing custom layers
# 
# The best way to implement your own layer is extending the tf.keras.Layer class and implementing:
# - `__init__`, where you can do all input-dependent initialization
# - `build`, where you know the shapes of the input tensors and can do the rest of the initialization
# - `call`, where you do the forward computation
# 
# Note that you don't have to wait until `build` is called to create your variables, you can also create them in `__init__`. Howeverr, the advantage of creating them in build is that it enables late variable creation based on the shape of the inputs the layer will operate on on. On the other hand, creating varriables in `__init__` would mean that shapes required to create the variables will  need to be explicitly specified.

# In[ ]:


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
        
    def build(self, input_shape):
        self.kernel = self.add_variable("kernel", shape=[int(input_shape[-1]), self.num_outputs])
        
    def call(self, input):
        return tf.matmul(input, self.kernel)
    
layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.variables)


# Note that you don't have to wait until `build` is called to create your variables, you can also create them in `__init__`.
# 
# Overall code is easier to read and maintail if it used standard layers whenever possible, as other readers will be familiar with the behavior of standard layers. If you want to use a layer which is not present in tf.keras.layers or tf.contrib.layers, consider filing a [github issue](https://github.com/tensorflow/tensorflow/issues/new) or, even better, sending us a pull request!
# 
# ## Models: composing layers
# 
# Many interesting layer-like things in machine learning models are implemented by composing existing layers. For example, each residual block in a resnet is a composition of convolutions, batch normalizations, and a shortcut.
# 
# The main class used when creating a layer-like thing with contains other layers is `tf.keras.Model`. Implementing one is done by inheriting from `tf.keras.Model`.

# In[ ]:


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters
        
        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()
        
        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()
        
        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn22c = tf.keras.layers.BatchNormalization()
        
    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2c(x)
        x = self.bn22c(x, training=training)
        
        x += input_tensor
        return tf.nn.relu(x)
    
block =  ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.variables])


# Much of the time, however, the models which composee many layers simply call one layer after the other. This can be done in very little code using `tf.keras.Sequential`

# In[ ]:


my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
                                 tf.keras.layers.BatchNormalization(),
                                 tf.keras.layers.Conv2D(2, 1, padding='same'),
                                 tf.keras.layers.BatchNormalization(),
                                 tf.keras.layers.Conv2D(3, (1, 1)),
                                 tf.keras.layers.BatchNormalization()])

my_seq(tf.zeros([1, 2, 3, 3]))


# **Next steps**
# 
# Now you can go back to the previous noteebook and adapt the linear regression example to use layers and models to be better structured.
