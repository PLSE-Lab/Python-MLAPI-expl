#!/usr/bin/env python
# coding: utf-8

# You'll try to build a neural network that predicts the price of a house according to a simple formula.
# 
# So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
# 
# How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.
# 

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow import keras


# In[ ]:



def house_model(y_new):
    
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0], dtype = float)
    ys = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 550.0], dtype = float)
    
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape = [1])])

    model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
    
    model.fit(xs, ys, epochs = 500)
    
    return model.predict(y_new)[0]


# In[ ]:


prediction = house_model([7.0])
print(prediction)


# In[ ]:




