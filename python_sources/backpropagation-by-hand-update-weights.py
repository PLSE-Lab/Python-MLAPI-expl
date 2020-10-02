#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from IPython.display import Image
import os


# This is a fully connected neural network. I am demonstrating the forward pass, backpropagation and the weight update process
# 
# matrix multiplication
# 
# H1=[x1, x2 ] [w1, w3]T
# H2=[x1, x2 ] [w2, w4]T
# 
# x represents input data, it can be a tokenized words, stock price etc.
# 
# - the weights are assigned in Keras, they are updated via backpropagation
# - used qudratic cost funtion for simplicity
# - bias = 0 for simiplicty
# - there are two ways of assigning weights in Keras in a dense layer. 'Random_uniform' or 'zeros'
# 
# model.add(Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros'))
# 

# In[ ]:


Image("../input/20200105_233950.jpg")


# - forward pass to get the values of the hidden layer, input the value into the sigmoid fucntion to get the output for hidden layer
# - use the hidden layer output to continue to forward pass to get the value of the output node (y)

# In[ ]:


Image("../input/20200105_234013.jpg")


# - input the y out value into the cost fucntion, the cost function takes into account both the output y1 and output y2
# - the goal is to minimized the cost funtion for a more accurate prediction
# - in this case output y1 predicts value of 0.6 for a target in reality that is 0 and output y2 predicts value of 0.65 for a target in reality that is 1
# 
# backpropagation step
# 
# - find the partial derivative of w5 w.r.t to the Error total ( the quadratic cost fucntion)
# - first component is the derivative of the cost function w.r.t to the contribution of output y1. Here use the chain rule
# - second component is the derivative of the output y1 (after input into the sigmoid) w.r.t the y1 (the value before input into the sigmoid function). We can prove that the sigmoid function in the binary class is y(1-y)
# 
# - third component is the derivative of the y1 (the value before input into the sigmoid function) w.r.t to w5, this is a linear function, there for the derivative of w5 it's itself

# In[ ]:


Image("../input/20200105_234028.jpg")


# - now we can update the original randomly initiated weight of 0.5 with the gradient w5 we just calculated w.r.t Error total
# - now w5 is updated. Keep in mind the learn rate; If the learn rate is too small or too large it can overshoot or become stuck at a local minimum

# In[ ]:


Image("../input/20200105_234036.jpg")


# - calculate and update w6, w7, w8
# - as an example we are going to update w6
# - the reason why we updated w6 is because we need continue to backpropagate to the weight of the input( first layer)
# - as an example we are going to update w1, which is a component of w5 and w6

# - just like before the question that we are asking is what is the closet component w.r.t the cost function that we are trying to find the gradient (rate of change of the weight)
# - that component is cloest at H1 because w1 is the closest to it
# - therefore first componet is the derivative of the cost function w.r.t to the contribution of H out 1.
# - second component is the derivative of the H out 1 (after input into the sigmoid) w.r.t the H1 (the value before input into the sigmoid function). We can prove that the sigmoid function in the binary class is y(1-y)
# 
# - third component is the derivative of the H1 (the value before input into the sigmoid function) w.r.t to w1, this is a linear function, therefore the derivative of w5 it's itself

# - first component is further broken down into two components, the first one coming from w5 and second from w6
# - the first component coming from w5 is actually passed down from what we calculated before (that's why backpropagation is so fast), the only change is that instead of original w5 it's the w5 new that we previously calculated
# 
# 

# In[ ]:


Image("../input/20200105_234044.jpg")


# - now we calculate the second Error component w.r.t H out 1, if you look at the setup from page 1, you can see that half of the Error ('half of error' w.r.t H out 1) is coming from w6 which in turn is coming from H out 1
# 
# - this is why w6 was previously caculated and updated to w6 new
# 

# In[ ]:


Image("../input/20200105_234105.jpg")


# - now that we have all 3 components, we can update w1

# In[ ]:


Image("../input/20200105_234122.jpg")


# - the update is very small because of random weight initialization
# - as you can see if the weight is not in the range of [0.3, 0.7] for example, then the slope (rate of change is close to 0) and when you pass through multiple layers the update becomes very small
# - thus other activation function of such are relu or leaky relu to mitigate this issue

# - now that the backpropagation process is understood, it will become the foundation mechanisms to comprehend more complex neural network architecture

# In[ ]:


Image("../input/20200105_234149.jpg")

