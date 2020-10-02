#!/usr/bin/env python
# coding: utf-8

# ### Session 04:
# 
# **Backpropagation**: Algorithm to caculate gradient  for all the weights in the network with several weights. 
# 
# * It uses the `Chain Rule` to calcuate the gradient for multiple nodes at the same time. 
# * In pytorch this is implemented using a `variable` data type and `loss.backward()` method to get the gradients

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# #### Using .backward method for linear regression
# 
# $y = x * w$
# 
# $loss =(\hat{y}-y)^2$

# In[ ]:


# Input data

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# In[ ]:


w = torch.tensor([1.0], requires_grad=True)


# In[ ]:


def forward(x):
    return x*w


# In[ ]:


def loss(y_pred,y):
    return (y_pred-y)**2


# In[ ]:


# Training loop

print('Predict (before training)', 4, forward(4).item())

# Training loop

for epoch in range(10):
#     l_sum=0
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # Forward pass
        l = loss(y_pred,y_val) # Loss
        l.backward() # Backpropagation
        print("\tgrad: ", x_val, y_val, w.grad.item())
        w.data = w.data - 0.01 * w.grad.item()

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()
        
    print(f"Epoch: {epoch} | Loss: {l.item()} | w: {w.item()}")
#     w_list.append(w)
#     mse_list.append(l_sum/3)
    
    
print('Predict (After training)', '4 hours', forward(4).item())    


# #### Using backward propagation for quadratic model
# 
# $\hat{y} = x^2*w_{2} + x*w_{1} + b$
# 
# $loss = (\hat{y}-y)^2$
# 
# * Using Dummy values of x and y
# 
# `x = 1,2,3,4,5`
# `y = 1,6,15,28,45`

# In[ ]:


x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [1.0, 6.0, 15.0, 28,45]


# In[ ]:


plt.plot(x_data,y_data)
plt.show()


# In[ ]:


# Initialize w2 and w1 with randon values

w_1 = torch.tensor([1.0], requires_grad=True)
w_2 = torch.tensor([1.0], requires_grad=True)


# In[ ]:


# Quadratic forward pass based on the function above. Taking b as zero for now

def quad_forward(x):
    return w_1*(x**2)+w_2*x


# In[ ]:


# Loss fucntion as per the defination above

def loss(y_pred,y):
    return (y_pred-y)**2


# In[ ]:


# Training loop

print('Predict (before training)', 6, quad_forward(6).item())

for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = quad_forward(x_val)
        l = loss(y_pred, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w_1.grad.item(), w_1.grad.item())
        w_1.data = w_1.data - 0.0012*w_1.grad.item()
        w_2.data = w_2.data - 0.0012*w_2.grad.item()
        
        # Manually zero the gradients after updating weights
        w_1.grad.data.zero_()
        w_2.grad.data.zero_()
        
    print(f"Epoch: {epoch} | Loss: {l.item()} | w1: {w_1.item()} | w2: {w_2.item()}")


print('Predict (After training)', 6, quad_forward(6).item())


# In[ ]:




