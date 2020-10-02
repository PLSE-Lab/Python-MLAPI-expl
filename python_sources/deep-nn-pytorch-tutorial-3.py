#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/anandsubbu007/Pytorch-Tutorial-Beginner/blob/master/Deep_NN_pytorch_3_.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # PyTorch Tutorials
# 
# 
# PyTorch is the premier open-source deep learning framework developed and maintained by Facebook.
# 
# At its core, PyTorch is a mathematical library that allows you to perform efficient computation and automatic differentiation on graph-based models. Achieving this directly is challenging, although thankfully, the modern PyTorch API provides classes and idioms that allow you to easily develop a suite of deep learning models.
# 
# 1. [PyTorch Tutorial - 1 (Basic)](https://www.kaggle.com/anandsubbu007/pytorch-basics-tutorial-1)
# 2. [PyTorch Tutorial - 2 (Autograd)](https://www.kaggle.com/anandsubbu007/pytorch-autograd-tutorial-2)
# 3. [PyTorch Tutorial - 3 (Deep Neural Network)](https://www.kaggle.com/anandsubbu007/deep-nn-pytorch-tutorial-3)
# 4. [PyTorch Tutorial - 3 (CNN-CIFAR10)](https://www.kaggle.com/anandsubbu007/cnn-cifar10-pytorch-tutorial-4)

# # Neural Network using PyTorch

# ![1.png](attachment:1.png)

# Fully Connected Neural Network
# 
#     Input Layer  --->  Hidden Layer   ---> Sigmoid Neuron
#       10 Units            5 Units               1 Units
# 

# ![2.jfif](attachment:2.jfif)

# ![3.jfif](attachment:3.jfif)

# ## torch.nn in pytorch
# PyTorch provides the torch.nn module to help us in creating and training of the neural network
# 
# [To know more torch.nn Class & Module ](https://www.javatpoint.com/torch_nn-in-pytorch)

# In[ ]:


# Import Pytorch & torch.nn Library
import torch
import torch.nn as nn


# In[ ]:


n_ip = 10 # Input Layer
n_hd = 5  # Hidden Layer
n_op = 1  # Output Layer
b_sz = 10 # batch size
epoch = 25 


# In[ ]:


x = torch.randn(b_sz, n_ip)
y = torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0]])


# In[ ]:


# building Model
model = nn.Sequential(nn.Linear(n_ip,n_hd),
                      nn.ReLU(),
                      nn.Linear(n_hd,n_op),
                      nn.Sigmoid())


# In[ ]:


#Defining Loss Function
loss_func = nn.MSELoss()
#Optimizer
opt = torch.optim.SGD(model.parameters(), lr = .01) #lr --> learning rate


# In[ ]:


for i in range(epoch):
  y_pred = model(x)
  loss = loss_func(y_pred,y)  

  print("epoch : ", i+1 , " loss : ", loss.item())
  loss.backward()
  
  #Update the parameters
  opt.step() #Updates all the parameters
  
  #Zero Gradients
  opt.zero_grad()
  
  #This will execute until total number of epochs


# # Custom NN Module

# With the use of torch.nn.module we can combine many simple layers to implement complex neural networks.
# 
# In other words, we can use it to represent an arbitary function f in Pytorch

# We subclass torch.nn.module for complex networks.
# 
# We override methods under --> torch.nn.module class
# 
#   1. __init__ function :
#         - invoked when we create instance of nn.Module
#          
#   2. forward function:
#         - We define how output will be computed

# In[ ]:


class custom_mod(nn.Module):
  def __init__(self,inp,hid,out):
    super(custom_mod,self).__init__()
    self.model = nn.Sequential(nn.Linear(inp,hid),
                               nn.ReLU(),
                               nn.Linear(hid,out),
                               nn.Sigmoid())
  def forward(self,x):
    return self.module(x)


# In[ ]:


cust_nn = custom_mod(n_ip,n_hd,n_op)


# In[ ]:


#Defining Loss Function
loss_func = nn.MSELoss()
#Optimizer
opt = torch.optim.SGD(model.parameters(), lr = .01) #lr --> learning rate


# In[ ]:


for i in range(epoch):
  y_pred = model(x)
  loss = loss_func(y_pred,y)  

  print("epoch : ", i+1 , " loss : ", loss.item())
  loss.backward()
  
  #Update the parameters
  opt.step() #Updates all the parameters
  
  #Zero Gradients
  opt.zero_grad()
  
  #This will execute until total number of epochs

