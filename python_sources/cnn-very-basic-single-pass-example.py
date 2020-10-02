#!/usr/bin/env python
# coding: utf-8

# **Simple CNN**
# 
# A simple convolution operation is demonstrated for the very beginners.
# 
# The idea of convolution comes from math.
# 
# In math, a convolution operation that slides two functions over each other, multiplying them.
# 
# Animation:
# [https://upload.wikimedia.org/wikipedia/commons/6/6a/Convolution_of_box_signal_with_itself2.gif]

# In[ ]:


import math
import numpy as np
from numpy import linalg
import scipy.io.wavfile as wav
from scipy.optimize import curve_fit
from scipy import special
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import wave
import csv
import random
import time
import os
import zipfile



print('library test ok')


# Didn't the get hang of Tensorflow 2.0 yet. So disabling eager execution.

# In[ ]:


tf.compat.v1.disable_eager_execution()


# Choosing in incredibly simple.
# The data would be the [https://en.wikipedia.org/wiki/Identity_matrix]
# The filter would be one of the [https://en.wikipedia.org/wiki/Dirac_spinor]
# 
# Let's see what a single pass convulation gives

# In[ ]:


x  = [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
y  = [[1.,0.],[0.,-1.]]
x1 = tf.constant( x  , dtype=tf.float32 )
y1 = tf.constant( y , dtype=tf.float32 )
x2 = tf.reshape(x1, [1, 3, 3, 1])
y2 = tf.reshape(y1, [2, 2, 1, 1])
z = tf.nn.conv2d(x2, y2, [1, 1, 1, 1], "VALID")
zout = tf.reshape(z, [2,2])
with tf.compat.v1.Session() as sess:
  sess.run( tf.compat.v1.global_variables_initializer() )
  print(sess.run(zout))


# If you're a beginner, I hope you now understand what a convolution operation and begin your projects in computer vision, image and speech recognition. 
# I might add CNN's aren't really useful anywhere else, even in speech synthesis.
# 
# This post was originally posted on [https://www.quora.com]. Read it here [https://www.quora.com/q/wueohrivrqkdezxx/Introductory-Convolutional-Neural-Networks]
# 
# If you want to disagree with that post or **any other post** from my quora profile [https://www.quora.com/profile/Marceline-Hosenback] you may comment on this post and **only on this post**.
# 
# Please keep in mind [https://www.kaggle.com/terms] and [https://www.kaggle.com/privacy]
