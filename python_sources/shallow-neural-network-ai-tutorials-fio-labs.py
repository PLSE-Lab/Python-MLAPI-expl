#!/usr/bin/env python
# coding: utf-8

# ## *Welcome to FIO Labs*
# 
# # This Tutorial is also available in Video format on our [YouTube Channel - FIOLabs](https://www.youtube.com/channel/UC6Vn_nUJeJ7PrvsayHPVOag) . Follow for more AI | ML | DL Tutorials.
# 
# # Shallow Neural Network
# 
# We've done coding for a simple [Perceptron](https://www.kaggle.com/prashfio/perceptron#Structure-of-Perceptron)  in our last tutorial using just *numpy*. Let's build a Shallow Neural Network *from the scratch*, i.e., a Neural Network with one input layer, one hidden layer, and one output layer.

# # Structure of Shallow Neural Network
# 
# Let's see how a Shallow Neural Network looks like.

# In[ ]:


from IPython.display import Image
import os
get_ipython().system('ls ../input/')


# We have SNN (Shallow Neural Network) and ReLU function, which we will come back to later. Let's look at SNN.

# In[ ]:


Image('../input/pictures1/snn.png')


# For simplicity's sake, we are taking only **2 inputs** plus a default bias. In **total**, there are **9 weights** of which 6 correspond to the inputs and the other 3 are of bias. In the hidden layer, we have two nodes, namely, **z_11 and z_12**. These are the summation functions in the hidden layer which will then be fed into activation functions, respectively, **a_11 and a_12**. The output from these activation functions will again have different weights and have the final summation at **z_2**. This z_2 will again be fed into activation function which yields us the **final output**. 
# 
# Let's start coding.

# In[ ]:


#Import numpy for numerical calculations

import numpy as np


# # 1. Randomly Initialize Weights.

# In[ ]:


input_weights = np.around(np.random.uniform(-5,5,size=6), decimals=2)
bias_weights = np.around(np.random.uniform(size=3), decimals=2)


# Let's see the random weights which were assigned to the inputs and the biases.

# In[ ]:


print(input_weights)
print(bias_weights)


# # 2. Assign values to inputs.

# By default, value for biases is equal to 1.

# In[ ]:


x_1 = 0.5 #input 1
x_2 = 0.82 #input 2

print('Input x1 is {} and Input x2 is {}'.format(x_1,x_2))


# # 3. Calculate linear combination of inputs.

# Let's calculate linear combination of inputs and their weights which will be assigned to **z_11**, the first node in the hidden layer.

# In[ ]:


z_11 = x_1 * input_weights[0] + x_2 * input_weights[1] + bias_weights[0]

print('The linear combination of inputs at the first node of the hidden layer is {}'.format(z_11))


# In[ ]:


z_12 = x_1 * input_weights[2] + x_2 * input_weights[3] + bias_weights[1]

print('The linear combination of inputs at the second node of the hidden layer is {}'.format(z_12))


# # 4. Calculate Output of Activation Function.
# 
# Now that we're done with the summation, let's feed this into the activation functions. We're taking ReLU as our activation function for this hidden layer.
# 
# Let's first visualize ReLU function and look at the formula.

# In[ ]:


Image('../input/pictures1/relu.png')


# As you can see from the above formula, it's clear that **ReLU excludes any x values which are less than 0 and activates only when x values are greater than 0**.
# 
# Let's compute the output of this activation function when z_11 is fed into it.

# In[ ]:


a_11 = max(0.0, z_11)

print('The output of the activation function at the first node of the hidden layer is {}'.format(np.around(a_11, decimals=4)))


# In[ ]:


a_12 = max(0.0, z_12)

print('The output of the activation function at the second node of the hidden layer is {}'.format(np.around(a_12, decimals=4)))


# # 5. Repeat the steps until you get your final output.

# Now, these outputs serve as the inputs to the Output Layer. So, let's repeat the last few steps in assigning weights and computing linear combination of inputs and then passing that into the activation function.

# In[ ]:


z_2 = a_11 * input_weights[4] + a_12 * input_weights[5] + bias_weights[2]

print('The linear combination of inputs at the output layer is {}'.format(z_2))


# Here, we will be feeding this summation into a non-linear activation function known as sigmoid function which is best suited for Output Layer.

# In[ ]:


Image('../input/pictures1/sigmoid.png')


# In[ ]:


y = 1.0 / (1.0 + np.exp(-z_2))

print('The output of the network for the given inputs is {}'.format(np.around(y, decimals=6)))


# In[ ]:




