#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/anandsubbu007/Pytorch-Tutorial-Beginner/blob/master/Pytorch_Autograd_2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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

# # Autograd

# ## Differential in Pytorch

# In[ ]:


#Importing Libraries
import torch
# Assign Device 
cuda0 = torch.device("cuda:0")


# In[ ]:


x = torch.tensor([5],dtype=torch.float32,requires_grad=True)
y = torch.tensor([6],dtype=torch.float32,requires_grad=True)
print(x)
print(y)


# In[ ]:


#defining the function
z = ((x**2)*y) + (x*y)
print(z)


# In[ ]:


#Using autograd
# Autograd to be applied on Scalars
total = torch.sum(z) # Converting to scalar
total


# In[ ]:


print(x.grad,y.grad)


# In[ ]:


total.backward() # to call grad function we need to call .backward() if not it will show as None


# In[ ]:


print("Def with resp. to  x   :",x.grad)
print("Def with resp. to  y   :",y.grad)


# Finding Deferintial for x & y
# 
#     z = (x^2)*y + xy
#       = (5^2 * 6) + 5*6
#       = 180
# 
#     dz/dx = 2xy + y
#           = (2 * 5 * 6) + 6
#           = 66
# 
#     dz/dy = x^2 + x
#           = 5^2 + 5
#           = 30

# # Implementing Autograd

# In[ ]:


x = torch.randint(-100,100,(100,), dtype = torch.float32 , device = cuda0)
y = (1.32*x) + 25                       # y = (w*x) + b     we are going to predict w & b


# In[ ]:


w = torch.ones(1,requires_grad = True, device = cuda0 )
b = torch.ones(1,requires_grad = True, device = cuda0 ) 
y_hat = (w*x) + b

epochs = 10000
lr = 0.000001


# In[ ]:


count = 0
for i in range(epochs):
  loss = torch.sum((y_hat - y)**2) 
  loss.backward() 
  #w -= lr*w.grad --> this will be considered as relationship
  with torch.no_grad(): # this will switch off gradients

    w -= lr*w.grad
    b -= lr*b.grad
    count += 1
    #setting gradients to be zero
    w.grad.zero_()
    b.grad.zero_() 
  
  y_hat = (w*x ) + b

print(count)
print("Predicted w value  :",w.item())
print("Predicted b value  :",b.item())

