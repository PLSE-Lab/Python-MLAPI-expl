#!/usr/bin/env python
# coding: utf-8

# I have recently started to learn PyTorch. It would be wise to share my learnings in a concise manner for anyone who is wishing to get started with PyTorch. My learning is borrowed from Official Documentation of PyTorch, blogs, and my own additions from my prior learnings and experience in TF. <br><br>
# I'll try to keep the notebook as much informative and relevant as possible. Hope that you find the notebook useful. <br>
# <br>
# Links to the upcoming notebooks will be added/updated here.
# <br>
# <br>
# ***If you like my work, please appreciate the same my giving an upvote to the notebook.***

# In[ ]:


# Import pytorch and numpy
import torch
import numpy as np


# # Tensor Initialization

# In[ ]:


a = torch.randint(low = 0, high = 10, size = (1,))
print(a)


# In[ ]:


print(type(a))


# In[ ]:


b = torch.tensor(5)
print(b)
print(type(b))


# In[ ]:


# Adding two tensors

print(a + b)
print(torch.add(a,b))


# In[ ]:


# Subtraction

print(a - b)
print(torch.sub(a,b))


# In[ ]:


# Absolute Difference
print(torch.abs(a - b))

# or

print(torch.abs(torch.sub(a,b)))


# # Basic Mathematical Operations using Torch Tensors (Matrices)

# In[ ]:


a = torch.randn((3,3))
print(a)


# In[ ]:


b = torch.randn((3,3))
print(b)


# In[ ]:


# MATRIX OPERATIONS


# In[ ]:


# Addition
print(a + b)
print('\n', torch.add(a,b))


# In[ ]:


# Subtraction

print(a - b)
print('\n', torch.sub(a,b))


# In[ ]:


# MATRIX MULTIPLICATION

print('\n', torch.mm(a,b))


# In[ ]:


# ALSO Matrix Multiplication

print(a @ b)


# In[ ]:


# Dot product
np.dot(a,b)


# In[ ]:


# element-wise multiplication

a * b


# In[ ]:


# Transpose operation

np.transpose(a)


# In[ ]:


# Also transpose operation
torch.t(a)


# In[ ]:


# concatenation of tensors


# In[ ]:


#row-stacking, i.e., axis = 0
torch.cat((a,b))


# In[ ]:


#column-stacking, i.e., axis = 1
torch.cat((a,b), axis = 1)


# In[ ]:


# reshaping of tensors

#while reshaping ensure the new dimensions product maintains the element count
print(a.reshape(1,9))


# In[ ]:





# # **Using PyTorch's AutoGrad Module**

# ![Gradient Graph](https://blog.paperspace.com/content/images/2019/03/image-4.png)

# Figure: Example of PyTorch Gradient Graph Calculation

# In[ ]:


# tensor without autograd

a = torch.rand(3,3)
print(a)


# In[ ]:


# tensor with autograd

a = torch.rand(3,3, requires_grad = True)
print(a)


# In[ ]:


# Let's begin with a simple linear equation y = x+ 5

x = torch.ones(3,3, requires_grad = True)
y = x + 5
print(y)


# In[ ]:


print(y.grad_fn)


# In[ ]:


print(y.requires_grad)


# In[ ]:


# Gradients and Backpropagation

x = torch.ones(2,2, requires_grad = True)
y = x + 3
z = y**2

res = z.mean()
print(z)
print(res)


# In[ ]:


# backpropagate and print the gradients
res.backward()
print(x.grad)


# Explantation - We begin with a 2x2 matrix named x with requires_grad set to True. Add 3 to every element of x to obtain y. A tensor z is obtained which will be the square of values of y. Z's mean is calculated and stoed in res. <br>
# y -> [4 4 <br>
#     4 4]
#      <br><br>
#      
# z -> [16 16 <br>
#      16 16] <br><br>
#      
# => z = (x + 3)^2 , for each x
# 
# res -> [16]
# 
# => res -> 1/4 * Sigma(z)
# 
# When you perform, x.grad, it is calculating the partial derivative of the equation above with respect to x.
# 

# By default, PyTorch expects backward() to be called for the last output of the network - the loss function. <br>
# **backward() is called on a scalar tensor and expects no arguments.**

# In[ ]:





# # Few More things about PyTorch Tensor

# Playing around with some data types

# In[ ]:


torch.zeros([2,4],dtype = torch.int32)


# In[ ]:


torch.ones([2,4], dtype = torch.float64)


# indexing elements in tensors

# In[ ]:


x = torch.tensor([[1,2,3], [4,5,6]])
print(x[1][2])


# Converting Numpy Arrays to Torch format and vice-versa

# In[ ]:


# Numpy to Torch format
array = np.arange(1,11)
tensor = torch.from_numpy(array)
print(tensor)


# In[ ]:


# Torch to numpy

print(torch.Tensor.cpu(tensor).detach().numpy())
print(type(torch.Tensor.cpu(tensor).detach().numpy()))


# *** The upcoming post will be on Logistic Regression and Neural Network basics using PyTorch. Stay Tuned!***

# **If you liked my kernel, please give an UPVOTE to appreciate my work and comment down below if you would like to add anything or suggest something.**
