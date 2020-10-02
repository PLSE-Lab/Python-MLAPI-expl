#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
Sequential = tf.keras.models.Sequential
Model = tf.keras.models.Model
layers = tf.keras.layers
Input = tf.keras.Input


# In[ ]:


# in the functional api you directly manipulate tensors and use layers as functions that take tensors and return tensors (hence the name funtional API)
input_tensor = tf.keras.Input(shape=(32,))           # a tensor
dense = tf.keras.layers.Dense(32, activation='relu') # a layer is a function
output_tensor = dense(input_tensor)                  # a layer may be called on a tensor, and it returns a tensor


# #Compare Sequential model and Functional model

# Sequential

# In[ ]:


seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))
seq_model.summary()


# In[ ]:


SVG(model_to_dot(seq_model,show_shapes=True).create(prog='dot', format='svg'))


# Functional

# In[ ]:


input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
func_model = Model(input_tensor, output_tensor)
func_model.summary()


# In[ ]:


func_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))

func_model.fit(x_train, y_train, epochs=10, batch_size=128)


# In[ ]:


score = func_model.evaluate(x_train, y_train)
score


# In[ ]:


SVG(model_to_dot(func_model,show_shapes=True).create(prog='dot', format='svg'))

