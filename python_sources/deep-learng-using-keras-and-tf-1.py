#!/usr/bin/env python
# coding: utf-8

# # Deep Learning using Keras & Tensorflow
# 
# ## Part - 1
# 
# ### Priya Ranjan Mohanty

# Implementing ANN ( Artificial Neural Network ) from scratch is very difficult as well as time consuming exercise.
# 
# So as to minimize the complexity & time needed to build an ANN , we have various Frameworks available .
# 
# So , what exactly is a Deep Learning Framework -
# 
# > A deep learning framework is an interface, library or a tool which allows us to build deep learning models more easily and quickly, without getting into the details of underlying algorithms. They provide a clear and concise way for defining models using a collection of pre-built and optimized components.
# 

# Instead of writing hundreds of lines of code from scratch, we can use a suitable framework to help us to build such a model quickly. 
# 
# Below are some of the key features of a good deep learning framework:
# 
# * Optimized for performance
# * Easy to understand and code
# * Good community support
# * Parallelize the processes to reduce computations
# * Automatically compute gradients

# Various Deep Learning Frameworks - 
# 
# * Tensorflow 
# * Keras
# * Pytorch
# * Caffe
# * Deeplearnig4j

# ### Tensorflow :
# 
# * Developed by Google Brain Team .
# * Available as Open Source since Nov 2015
# * TensorBoard for effective Data Visualization 
# * Written in C++ & Python
# * Has a large Community Base 
# 
# 
# ### Keras 
# 
# * Developed by Francois Chollet , Google Engineer 
# * High Level Library & hence enables fast experimentation 
# * Uses Tensorflow ( primarily ) as backend 
# * Easiest Deep Learning Framework for beginners
# 
# 
# ### Pytorch
# 
# * Developed by Facebook's AI research group
# * Easy to use when compared to Tensorflow 
# * Written in C & Python
# * Not regularly used as a Production framework
# * 
# 
# ### Caffe
# 
# * Developed by Berkeley AI Research ( BAIR ) 
# * Very quick image processing 
# * Not effective for RNNs & Language processing 
# * Used for deploying models on Smart Devices 
# * Written in C++
# 
# ### Deeplearning4j
# 
# * Developed by Adam Gibson , Skymind 
# * it is a Java Library for Deep Learning 
# * Takes advantage of distributed frameworks ( spark , hadoop ) 
# * can process huge amount of data without sacrificing speed 

# ### Lets write a small program in Tensorflow to add two numbers!

# In[ ]:


# import tensorflow
import tensorflow as tf


# In[ ]:


# build computational graph
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)


# In[ ]:


addition = tf.add(a, b)


# In[ ]:


# initialize variables
init = tf.initialize_all_variables()


# In[ ]:


# create session and run the graph
with tf.Session() as sess:
    sess.run(init)
    print("Addition: %i" % sess.run(addition, feed_dict={a: 2, b: 3}))


# In[ ]:


# close session
sess.close()

