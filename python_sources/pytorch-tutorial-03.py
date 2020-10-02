#!/usr/bin/env python
# coding: utf-8

# ## Session: 03
# 
# Gradient Descent
# 
# * How the weights get update to ensure we reach minimum loss.
# * This method is called Gradient Descent.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


# In[ ]:


w_list = []
mse_list=[]


# In[ ]:


# Input data

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0


# In[ ]:


# Function for forward pass to predict y
def forward(x):
    return x*w


# In[ ]:


# Function to calcuate the loss of the model
# Loss is the square of difference of prediction and actual value

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2


# #### Gradient Descent
# 
# We update the `w` such that loss is minimum. The factor by which `w` is updated is called `alpha(learning rate)`.
# 
# New `w` is `w` minus `alpha` times derivative of `loss` against `w`
# 
# $w=w-a*\frac{d(loss)}{dw}$
# 
# This equation is dependent on how the loss function has been defined. 
# In the current case below formula will dictate how to update the value of w for each pass. 
# 
# $w = w - a*2x(xw-y)$
# 

# In[ ]:


# Function to calcualte the gradient for w to be updated and get min loss.
# y_pred closer to y

# Gradient = derivative of the loss for constant x and y

# We are going to use a as 0.01 for starters

def gradient(x,y):
    return 2*x*(x*w-y)


# In[ ]:


# Training loop

print('Predict (before training)', 4, forward(4))

# Training loop

for epoch in range(100):
    l_sum=0
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w-0.01*grad
        print('\tgrad: ', x_val, y_val, grad)
        l=loss(x_val, y_val)
        l_sum+=l
        
    print('Progress: ', epoch, 'w=', w, 'loss=', l)
    w_list.append(w)
    mse_list.append(l_sum/3)
    
    
print('Predict (After training)', '4 hours', forward(4))    


# In[ ]:


plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()


# In[ ]:




