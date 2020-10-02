#!/usr/bin/env python
# coding: utf-8

# ## Imports
# 
# As we normally include our header files  in c/c+ code. Similarly, in python, we use "import" keyword. So, here we have imported the relevant packages which we are going to use in our model.
# 
# I encourage you to go to tensorflow website and browse to python api's for Tensorflow 2.0 an d check what all differnt parameters are supported by the functions mentioned in this model (like sequential, dense, fit, compile). Because ultimatley we need to refer to the documentation to implement any new thing 

# In[ ]:


import numpy as np # linear algebra
import tensorflow as tf
from tensorflow import keras


# ## Define the Model
# 
# The first step is to define the model to decide about how many layers we are going to use, how many neurons should be present in each layer, what will be the shape of our input data (in our example we have a 1D array)

# In[ ]:


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


# ## Compile the Model
# 
# Like how we need to compile our normal c/c++/java code. Similar to that we need to compile our model as well by doing some settings like in below line we have selected our optimizer as "stochastic Gradient Descent" and algorithm to calculate loss as "mean_squared_error(MSE)".
# 
# Since we are using frameworks like keras and tensorflow, so, all these maths behind sgd, MSE are implemented in tensorflow and we just need to pass the parameters to make it work.

# In[ ]:


model.compile(optimizer='sgd', loss='mean_squared_error')


# ## Provide the Labeled Data (Both Input and Output)
# In Supervised Learing, dataset is provided as both Inputs & their corresponding output. So, in the below data, the inputs are the x_train values and the outputs are y_train values. If we see the relationship between x and Y, we can see that y=2x+10.
# 
# This data has to be converted into an array format with floating point values and this is done by using numpy library.
# 
# In our normal programming in c/c++/java, we need to define this eqation "y=2x+1" in our code and then only our code will be able to calculate the value of y for any given x. But in Machine/Deep Learning, we let the model or algorithm to find the correlation between x and y and derive the formula of its own.
# 
# Since this is a very simple example, so only 7 datasets are taken to train the model, but in actual scenarios, tens of thousands of dataset is required.

# In[ ]:


x_train = np.array([-2, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5, 6, 7, 8], dtype=float)
y_train = np.array([6.0, 8.0, 10.0, 11.0, 14.0, 16.0, 18.0, 20, 22, 24, 26], dtype=float)


# ## Training the Neural Network
# Here you can see that with each Epoch completion, the loss is reducing. We need to keep an eye on the Loss because the loss should be decreasing with every Epoch cycle. When the Loss reaches to a very small value and stops decreasing further, then it means that the model is not going to train any further and its better to use that much Epoch's only (knownn as Early stopping) because otherwise the Loss may start increasing again as the model will think that it still needs to train and will try to find another global minimum. 
# 
# Training the model works by using the labeled dataset (Input, output) and try to apply its own formula to calculate the output and keeps on improvig the result thereby reducing the Loss. for eg. for the first time it creates a relationship that y = 2x+1 and then calculated the Loss. It then tried with y = 2x+5and check that the loss has decreased. So, after multiple Epoch's model could predict the final equation of y=2x+10.

# In[ ]:


model.fit(x_train, y_train, epochs=500)


# ## Test the Model
# Here we can see that we were assuming that the predicted output should be exactly equals to "y=2x+10" but instead of this, we got some variation in the result because we feed very little amount of training data to make the model. But still with this, the model performs so well. 
# 
# We can also try to train the model with more some of epoch's or re-run the training code to get better results.
# 
# You can try it by passing any value and check the result.

# In[ ]:


print(model.predict([30.0]))


# In[ ]:




