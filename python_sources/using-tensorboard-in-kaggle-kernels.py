#!/usr/bin/env python
# coding: utf-8

# # Using TensorBoard in Kaggle Kernels
# 
# **Note:** TensorBoard requires a running kernel, so its output will only be available in an editor session.

# In[1]:


# Load the extension and start TensorBoard

get_ipython().run_line_magic('load_ext', 'tensorboard.notebook')
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[2]:


# Create your model

import tensorflow as tf
mnist = tf.keras.datasets.mnist

((x_train, y_train), (x_test, y_test)) = mnist.load_data()
(x_train, x_test) = (x_train / 255.0, x_test / 255.0)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'],
)


# In[3]:


# Configure the TensorBoard callback and fit your model

tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")

model.fit(
  x_train,
  y_train,
  epochs=5,
  callbacks=[tensorboard_callback],
)

