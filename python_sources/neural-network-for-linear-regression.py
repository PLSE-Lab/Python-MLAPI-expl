#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This is a notebook Fahrenheit-celsuis prediction using simple neural networks in deep learning:

import tensorflow as tf
from tensorflow import keras
from keras import layers,Sequential,optimizers,losses


# In[ ]:


#Poor data:
celsius_q    = [-40,  0 ,  8, 15, 38  , 32  , 122 ]
fahrenheit_a = [-40, 32 , 46, 59, 100 , 89.6,251.6]


# In[ ]:


# Sufficient amount of data:
celsius_q    = [-40,-35,-30,-25,-20,-15, -10,-5 ,  0 ,1   ,5 ,  8,10, 15, 22,  38 , 32  , 60 ,122]
fahrenheit_a = [-40,-31,-22,-13,-4 ,-5 ,  14,-23, 32 ,33.8,41, 46,50, 59, 72, 100 , 89.6, 140,251.6]


# In[ ]:


# Simple neural network:
layer1 = tf.keras.layers.Dense(units = 1,input_shape=[1])


# In[ ]:


model = tf.keras.Sequential([layer1])
model.compile(loss ='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))


# In[ ]:


history = model.fit(celsius_q,fahrenheit_a, epochs= 3500)


# In[ ]:


print(model.predict([100]))


#     #answer should be 212

# In[ ]:


print("These are the layer variables: {}".format(layer1.get_weights()))


# In[ ]:


import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])


# In[ ]:


input_layer = tf.keras.layers.Dense(units = 4 , input_shape =[1])
hiddenlayer1= tf.keras.layers.Dense(4)
hiddenlayer2= tf.keras.layers.Dense(4)
hiddenlayer3= tf.keras.layers.Dense(4)
outputlayer = tf.keras.layers.Dense(1)


# In[ ]:


model = tf.keras.Sequential([input_layer,hiddenlayer1,hiddenlayer2,hiddenlayer3,outputlayer])


# In[ ]:


model.compile(loss='MSE',optimizer = tf.keras.optimizers.Adam(0.01))
hist = model.fit(celsius_q,fahrenheit_a, epochs= 1000)


# In[ ]:


model.predict([100])


# In[ ]:




