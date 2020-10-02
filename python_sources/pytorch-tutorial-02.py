#!/usr/bin/env python
# coding: utf-8

# ## Session: 02 
# 
# Linear model: $y=w*x$
# 
# * In this session we create a linear model for given `x_data` and `y_data`. 
# * Eventually we also plot the value of `w` against the difference in the prediction and actual value.  

# In[ ]:


import torch
import numpy as np


# In[ ]:


w_list=[]
mse_list=[]


# In[ ]:


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# In[ ]:


w = 1 # Random value
def forward(x):
    return x*w


# In[ ]:


def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)


# In[ ]:


for w in np.arange(0.0,4.1,0.1):
    print("w=", w)
    l_sum=0
    for x_val, y_val in zip (x_data, y_data):
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum+=l
        print("\t", x_val, y_val, y_pred_val, l)
        
    print("MSE=", l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()


# In[ ]:




