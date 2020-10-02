#!/usr/bin/env python
# coding: utf-8

# Import torch library

# In[ ]:


import torch


# Creating an empty tensor

# In[ ]:


x = torch.empty(5, 3)


# In[ ]:


print(x)


# Creating a tensor with random values

# In[ ]:


y = torch.rand(5, 3)
y


# Multiply two tensors

# In[ ]:


z = x * y
z


# Random 5x5x3 tensor. Like an input image.

# In[ ]:


img = torch.rand(5, 5, 3)
img


# Zero tensor 3x3

# In[ ]:


zero_tensor = torch.zeros(3, 3, dtype=torch.long)
zero_tensor


# Tensor of 1s - 3x3

# In[ ]:


ones = torch.ones(3,3, dtype = torch.long)
ones


# Moving ones tensor to GPU from CPU.

# In[ ]:


ones = ones.to("cuda")


# In[ ]:


print(ones)


# Standard approach.

# In[ ]:


if torch.cuda.is_available():
    zero_tensor = zero_tensor.to("cuda")


# In[ ]:


print(zero_tensor)


# In[ ]:


zero_tensor = zero_tensor.to("cpu")


# In[ ]:


print(zero_tensor)


# Let's experiment their time of computation on cpu and gpu. Creating two large tensors and multiplying them. First on CPU and then on GPU.

# In[ ]:


A = torch.rand(1024, 1024, 3)


# In[ ]:


B = torch.rand(1024, 1024, 3)


# In[ ]:


import time


# In[ ]:


get_ipython().run_cell_magic('time', '', 'C = A * B')


# Moving tensors to gpu.

# In[ ]:


A = A.to("cuda")
B = B.to("cuda")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'C = A * B')


# Time was taken more because result C is in CPU. Now, we create Z tensor with ones and move it to the gpu and then compute A * B. Computation takes place in gpu and result also stored in gpu.

# In[ ]:


Z = torch.ones_like(A)


# In[ ]:


Z = Z.to("cuda")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'Z = A * B')


# In[ ]:


Z.shape


# In[ ]:


T = torch.ones_like(Z)


# In[ ]:


print(T)


# In[ ]:


P = torch.randn_like(T, dtype=torch.float)


# In[ ]:


print(P[0][0])


# In[ ]:


print(P.size())


# In[ ]:


P.add_(T) # in-place addition


# ## Reshaping

# In[ ]:


x = torch.randn(4,4)


# In[ ]:


x


# In[ ]:


y = x.view(16,1)
y


# In[ ]:


y.size()


# In[ ]:


z = x.view(8,2)
z


# In[ ]:


g = x.view(-1, 8)
g


# In[ ]:


a = torch.randn(2,2)


# In[ ]:


print(a[0][0])
print(a[0][0].item()) #accessing the item from the tensor - works for scalar only


# ## Autograd
# Package in PyTorch for calculating gradients. Let's see!

# In[ ]:


x = torch.ones(2, 2, requires_grad=True)


# In[ ]:


print(x)


# In[ ]:


y = x + 2
print(y)


# In[ ]:


z = y * 3


# In[ ]:


o = z.mean()


# In[ ]:


print(z)
print(o)


# In[ ]:


o.backward()


# In[ ]:


print(x.grad)


# In[ ]:


print(y.grad)
print(z.grad)


# In[ ]:


x = torch.ones(1, 1, requires_grad=True)


# In[ ]:


y = x * 3
print(y)


# In[ ]:


z = y + 2
print(z)


# In[ ]:


z.backward()


# In[ ]:


print(x.grad)


# In[ ]:




