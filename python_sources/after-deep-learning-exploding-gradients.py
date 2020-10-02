#!/usr/bin/env python
# coding: utf-8

# # Exploding gradients 
# 
# 
# # What are exploding gradients? 
# For a video definition, including a recap on vanishing gradients, check out this video:
# https://www.youtube.com/watch?v=qO_NLVjD6zE 
# 
# Here is a summary of what it says:
# 
# Exploding gradients are the opposite of the vanishing gradient problem: the change in weights towards the start of the neural network are too large during backpropagation!
# By large changes, I mean changes that are greater than 1. 
# 
# The larger the terms are that are being multiplied together to create the gradient, the larger the gradient will be, hence exploding in size. 
# 
# So vanishing and exploding gradients are part of a more general problem: unstable gradients. 
#  
# In other words: exploding gradients can result in an unstable neural network that cannot learn.
# 
# # How do you know if you have exploding gradients?
# 
# * the model is unable to get traction on your training data (poor loss)
# * unstable model - large changes in loss from update to update
# * the model goes to NaN during training
# * the model weights quickly get large during training 
# * the model weights go to NaN during training 
# * the error gradient values are consistenly over 1 for each node and layer during training
# 
# # How do you fix exploding gradients? 
# Simply: weight initialization.  
#  
# The values for the weights are initially given random numbers - typically with mean 0 and standard deviation of 1. The variance of the node further down the neural network is the sum of all the variances of the neurons that are being fed into it. 
# 
# For example, if there are 200 nodes that have variance 1 and are being fed to a single node, then this node has variance 200, with standard deviation of sqrt(200). 
# 
# As the neuron has a large standard deviation, it is more likely that the value it takes is large too. When this large value is passed to the activation function (for example sigmoid pictured below) then most positive inputs will be mapped to one and most negative inputs will be mapped to zero. The outputs closer to 1 will be remembered. 
# 
# ![image.png](attachment:image.png) 
# Image taken from: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
# 
# During training updates there will only be small changes in the activation output, which means that the updates will be inefficient, hence not producing a good model. 
# 
# *But... how do we fix this issue?*
# 
# We set the variance of the weights feeding into the node to 1/n, so that the node has variance = (1/n + 1/n + ... + 1/n) = 1. 
# 
# To get these nodes to have a variance of 1/n, we multiply the weights that feed into the node by sqrt(1/n). This is called Xavier initialization. 
# 
# NOTE: if we are using ReLU as our initialization function then the variance of the weights should be set to 2/n, NOT 1/n, hence the weights should be multiplied by sqrt(2/n). 
# 
# There are other initialization techniques, but Xavier initialization is the most popular solution to solving exploding and vanishing gradient problem. 
# 
# Keras example: 

# In[ ]:


from keras.models import Sequential 
from keras.layers import Dense, Activation 

model = Sequential([
    Dense(16, input_shape=(1,5), activation = 'relu')
    Dense(32, activation = 'relu', kernel_initializer = 'glorot_uniform')
    Dense(2, activation = 'softmax')
])


# Setting 'kernel_initializer = 'glorot_uniform'' is the Xavier initialization using the uniform distribution. 'glorot_normal' sets the Xavier initialization as the normal distribution. 
# 
# However, this is the default for dense layers and other layers in Keras! So the problem is essentially solved for us. 
# 
# Here is a link to some more initializers: https://keras.io/api/layers/initializers/

# References: 
# 
# https://www.youtube.com/watch?v=qO_NLVjD6zE  
# 
# https://machinelearningmastery.com/exploding-gradients-in-neural-networks/#:~:text=In%20recurrent%20neural%20networks%2C%20exploding,long%20input%20sequences%20of%20data.&text=%E2%80%94%20On%20the%20difficulty%20of%20training%20recurrent%20neural%20networks%2C%202013.   
# 
# https://www.youtube.com/watch?v=8krd5qKVw-Q
