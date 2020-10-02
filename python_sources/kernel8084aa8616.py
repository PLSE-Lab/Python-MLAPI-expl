#!/usr/bin/env python
# coding: utf-8

# **Neural Network that predicts the price of a house according to a simple formula.**
# 
# > So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
# > 
# > How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
xs = np.array([2.0,3.0,4.0,5.0,6.0,8.0,9.0,10.0],dtype=float)
ys = np.array([150.0,200.0,250.0,300.0,350.0,450.0,500.0,550.0],dtype=float)
model.fit(xs,ys,epochs=500)
print(model.predict([7.0]))


# In[ ]:




