#!/usr/bin/env python
# coding: utf-8

# ### In this notebook, we will use a very simple neural network to make predictions.
# 
# #### Ref : https://www.youtube.com/watch?v=_Z9TRANg4c0

# In[ ]:


# Let's start by importing some useful libraries

import tensorflow as tf
import numpy as np
from tensorflow import keras


# In[ ]:


# Let's now create the simplest possible neural network i.e., with 1 layer and 1 neuron

model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])


# In[ ]:


# Let's compile our model by specifying an optimizer and loss function 

model.compile(optimizer = 'sgd', loss = 'mean_squared_error')


# In[ ]:


# Time to provide the data to our NN

X = np.array([-1,0,1,2,3,4,5,6,7,8,9,10], dtype = float)
Y = np.array([-10,0,10,20,30,40,50,60,70,80,90,100], dtype = float)


# In[ ]:


# Training the NN

model.fit(X,Y,epochs = 500)


# In[ ]:


# So now we have a model that is trained to learn the relationship between X and Y. Let's try to predict the value of 
# X which the model have not seen before

print(model.predict([25.0]))


# #### We were expecting Y to be 250 but out NN gave us the value closeby to 250. This is because NN deals with probabilities. With only 12 data points, our model calculated that there is a very high probability that the relationship between X and Y is Y = 10X. As a result, the prediction for 25 is very close t0 250 but not exactly 250.  
