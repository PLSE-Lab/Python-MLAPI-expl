#!/usr/bin/env python
# coding: utf-8

# **PREDICTING THE RELATIONSHIP BETWEEN VARIABLES**
# > 
# > *For instance*
# > float my_function(float x){
# >     float y = (3 * x) + 1;
# >     return y;
# > }

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow import keras


# In[ ]:



model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model.fit(xs, ys, epochs=100)

print(model.predict([20.0]))

