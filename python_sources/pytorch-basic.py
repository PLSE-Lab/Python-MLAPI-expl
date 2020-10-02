#!/usr/bin/env python
# coding: utf-8

# Tensors are similar to NumPy ndarray, with the addition being that Tensors can also be used on a GPU to accelerate computing.

# In[ ]:


from __future__ import print_function
import torch


# there are various ways to create the matrix using PyTorch as shown below,

# In[ ]:


x=torch.empty(5, 3)
print(x)


# Everytime I run the above command 'torch.empty' it will give different results as its uninitialised

# In[ ]:


x=torch.rand(5,3) # It will create a randomly initialized matrix
print(x)


# In[ ]:


x=torch.zeros(5,3) 
x
x1=torch.zeros(5,3,dtype=torch.long)# matrix with the datatype long
x1


# In[ ]:


x1


# In[ ]:


x=torch.tensor([5.5,3]) # crating tensor from the data, its a direct method
print(x)


# In[ ]:


# tensor can be created based on the existing tensor. The new command will use the same datatype unless it is defined newly.

x1=x.new_ones(5,3, dtype=torch.double)
x=torch.randn_like(x1, dtype=torch.float)
print(x)


# In[ ]:


print(x1)


# I have use the x1 for creating x. Here the datatype for the x is defined as float.

# In[ ]:


print(x.size())

torch.size is a tuple, it support all the tuple operation.
# In[ ]:


y=torch.zeros(5,3)
y
print(x+y) # it can also be written as print(torch.add(x,y))


# In[ ]:


print(torch.add(x,y))


# In[ ]:


print(x[3,]) # printing the 3rd row; (a,b), here a represent rows and b columns


# In[ ]:


#converting tensor into numpy and vice versa
a=torch.ones(5)
print(a)


# In[ ]:


b=a.numpy()
print(b)


# In[ ]:


a.add_(0.5) # adding 0.5 to a
print(a)
print(b)


# In[ ]:


import numpy as np
a=np.ones(5)
b=torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)


# In[ ]:


# moving tensor is very easy. It can be done by using .to method
if torch.cuda.is_available():
    device = torch.device("cuda")          
    y = torch.ones_like(x, device=device)  
    x = x.to(device)                    
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) 
    

