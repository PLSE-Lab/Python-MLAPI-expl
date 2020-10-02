#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np


# ## Basic  Operation

# In[ ]:


x = torch.empty(2,2,2)
print(x)


# In[ ]:


x = torch.rand(2,2)
print(x)


# In[ ]:


x = torch.zeros(2,2)
print(x)


# In[ ]:


x = torch.ones(2,2)
print(x)


# In[ ]:


# Set dtype for tensor
x = torch.ones(2,2, dtype= torch.double)
print(x.dtype)
print(x.size())


# In[ ]:


# Change a list into tensor
x = torch.tensor([2.5,0.1])
print(x)


# ## Tensor Operation

# In[ ]:


x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z = x+y
# z = torch.add(x,y)
print(z)


# In[ ]:


# Inplace addition
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
y.add_(x) # _ means that it will inplace value y
print(y)


# In[ ]:


# z = x-y
# z = x*y
# z = x/y
# z = torch.sub(x,y)
# z = torch.mul(x,y)
# z = torch.div(x,y)
# y.mul_(x)


# In[ ]:


# Slicing
x = torch.rand(5,3)
print(x)
print(x[1,:])
print(x[0,0].item()) # Retrieve just the value


# ## Reshape Tensor

# In[ ]:


x = torch.rand(4,4)
print(x)
y = x.view(16)
print(y)
print(y.size())


# In[ ]:


x = torch.rand(4,4)
print(x)
y = x.view(-1,8)
print(y)
print(y.size())


# ## tensor to numpy.ndarray

# In[ ]:


a = torch.ones(5)
print(a)
print(type(a))

# Change tensor into numpy.ndarray
b = a.numpy()
print(type(b))


'''
Cautious !
If they run in CPU, that means they share same memory location, so if you change 'a' values, b would also change !
'''


# In[ ]:


a = np.ones(5)
print(a)

b = torch.from_numpy(a)
print(b)


# ## Checking GPU and Switching GPU to CPU (viceversa)

# In[ ]:


if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.ones(5, device=device) # Operation in GPU
    y = torch.ones(5)
    y = y.to(device) # Operation in GPU
    z = x+y # Operation in GPU
    #z.numpy() # Will get error, because only handle CPU tensor
    # First convert it into CPU
    z = z.to('cpu')


# ## Set gradient

# In[ ]:


# Allow to calculate gradient for this tensor in optimization step
x = torch.ones(5, requires_grad = True)
print(x)


# In[ ]:




