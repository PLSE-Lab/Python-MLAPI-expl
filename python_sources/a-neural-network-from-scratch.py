#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This Notebook impliments the Neural Network to describe the Backpropogation. 

# ### This imports numpy, which is a linear algebra library. 

# In[ ]:


#importing numpy
import numpy as np


# A sigmoid function maps any value to a value between 0 and 1. We use it to convert numbers to probabilities. It also has several other desirable properties for training neural networks.
# Notice that this function can also generate the derivative of a sigmoid (when deriv=True). One of the desirable properties of a sigmoid function is that its output can be used to create its derivative. If the sigmoid's output is a variable "out", then the derivative is simply out * (1-out). This is very efficient. 

# In[ ]:


#sigmoid and darivative of sigmoid function
def sigdiv(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


# Inputs          Output<br>                                                                                                                                                                                                                                         
# A--B--C---D                                                                                                                                                                                                                                                   
# 0--0---1---0                                                                                                                                                                                                                                                   
# 1---1---1----1                                                                                                                                                                                                                                                   
# 1---0---1----1                                                                                                                                                                                                                                                     
# 0---1---1---0                                                                                                                                                                                                                                                       
# 

# It appears to be completely unrelated to column three, which is always 1. However, columns 1 and 2 give more clarity. If either column 1 or 2 are a 1 (but not both!) then the output is a 1. This is our pattern.( A XOR B)
# 
# This is considered a "nonlinear" pattern because there isn't a direct one-to-one relationship between the input and output. Instead, there is a one-to-one relationship between a combination of inputs, namely columns 1 and 2.

# In[ ]:


# input arrays
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])


# In[ ]:


# output array
Y = np.array([[0],
            [1],
            [1],
            [0]])


# In[ ]:


np.random.seed(0)


# In[ ]:


# Randomly initializing our weights with mean zero
W0 = 2*np.random.random((3,4)) - 1
W1 = 2*np.random.random((4,1)) - 1


# ## Training neral network
# This begins our actual network training code. This for loop "iterates" multiple times over the training code to optimize our network to the dataset.

# In[ ]:


for j in range(100000):
    #1 Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = sigdiv(np.dot(l0,W0))
    l2 = sigdiv(np.dot(l1,W1))
    
    #2 Calculating error
    l2_error = Y - l2
    
    #printing error
    if (j% 10000) == 0:
        print("Error iafter " + str(j) + " itration :" + str(np.mean(np.abs(l2_error))))
    
    # in what direction is the target value
    #3 calculating change to made in weights. 
    l2_delta = l2_error*sigdiv(l2,deriv=True)
    
    #4 how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(W1.T)
    
    # in what direction is the target value
    # calculating change to made in weights.
    l1_delta = l1_error * sigdiv(l1,deriv=True)
    
    #5 updating weights
    W1 += l1.T.dot(l2_delta)
    W0 += l0.T.dot(l1_delta)


# 1. Three layers neural network
# 
#    1.1 Input layer L0.                                                                                                                                                                                                                                         
#   
#   1.2 Hidden layer L1.
#    
#    1.2 Output layer L2 
#    .
# 2. Error calculation for hidden layer wights. l2_error is the amount hidden layer has missed.
# 
# 3. Calculating the change to made in W2 weights by finding the derivetive of L2.
# when L2_errors are multiplied by L2 derivative then the confident errors are muted beacuse L2_error will have close value to zero.
# 
# 4. Caculating the error passed by l1 to l2.
# 5. updating the weights by adding them(W1,W2) to their respective errors. 
# 
# The weights are updated at each itration of loop and neural network learns on each back propogation.
# Remember that the input is used as a single batch.
# 

# In[ ]:


#Output after training
l2


# 

# Created by :**MOHIT CHATURVEDI**
