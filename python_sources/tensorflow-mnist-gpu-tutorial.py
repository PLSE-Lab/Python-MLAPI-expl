#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# At present, you can choose from three popular open source Deep Learning libraries:
# 
# - TensorFlow, 
# - Microsoft Cognitive Toolkit (CNTK), 
# - Theano. 
# 
# Therefore, to avoid any confusion, we will refer to Keras implementation as multi-backend Keras.
# 
# TensorFlow itself now comes bundled with its own Keras implementation,tf.keras. It only supports TensorFlow as the backend.
# 
# ![image.png](attachment:image.png)
# 
# This short introduction uses Tensorflow Keras to:
# 
# - Build a neural network that classifies images.
# - Train this neural network.
# - And, finally, evaluate the accuracy of the model.
# 
# # Reference
# https://www.tensorflow.org/tutorials/quickstart/beginner

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf


# ## To Check GPU Availability in Tensorflow

# In[ ]:



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


# # Listing Devices including GPU's with Tensorflow

# In[ ]:



from tensorflow.python.client import device_lib

device_lib.list_local_devices()


# # To Check GPU in Tensorflow

# In[ ]:



tf.test.is_gpu_available()


# # Load MNiST Dataset

# In[ ]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# # Pre-processing of Training and Test Datasets

# In[ ]:


x_train, x_test = x_train / 255.0, x_test / 255.0


# # Create Sequential Model Using Tensorflow Keras
# 
# Architecture of the Network is :-
# 
# 1). Input layer for 28x28 images in MNiST dataset
# 
# 2). Dense layer with 128 neurons and ReLU activation function
# 
# 3). Output layer with 10 neurons for classification of input images as one of ten digits(0 to 9)

# In[ ]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


# In[ ]:


predictions = model(x_train[:1]).numpy()
predictions


# # Creating Loss Function

# In[ ]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# # Compile the Model Designed Earlier
# 
# Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
# 
# - Loss function 
# This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# 
# - Optimizer 
# This is how the model is updated based on the data it sees and its loss function.
# 
# - Metrics
# Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

# In[ ]:


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# # Training and Validation
# 
# The Model.fit method adjusts the model parameters to minimize the loss:
# 

# In[ ]:


model.fit(x_train, y_train, epochs=5)


# The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".

# In[ ]:



model.evaluate(x_test,  y_test, verbose=2)


# ### Please Upvote,Comment, Fork and Share to Help me with my efforts to help the community.
