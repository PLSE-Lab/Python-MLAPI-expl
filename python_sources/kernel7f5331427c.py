#!/usr/bin/env python
# coding: utf-8

# # PyTorch Basics: Tensors & Gradients

# In[ ]:


# import torch
import torch


# ## Tensors
# 
# At its core, PyTorch is a library for processing tensors. A tensor is a number, vector, matrix or any n-dimensional array. Let's create a tensor with a single number:

# In[ ]:


# Number
t1=torch.tensor(5.)
t1
# t1 = torch.tensor(4.)
# t1


# `4.` is a shorthand for `4.0`. It is used to indicate to Python (and PyTorch) that you want to create a floating point number. We can verify this by checking the `dtype` attribute of our tensor:

# In[ ]:


t1.dtype
# t1.dtype


# Let's try creating slightly more complex tensors:

# In[ ]:


# Vector
t2 = torch.tensor([1, 3, 4, 9.4, 4])
print(t2)
print(t2.dtype)

# t2 = torch.tensor([1., 2, 3, 4])
# t2


# In[ ]:


# Matrix
t3 = torch.tensor([[4,5],
                   [6,7],
                   [4,1],
                   [0,9],
                   [3,1]])

print(t3)
print(t3.dtype)

# t3 = torch.tensor([[5., 6], 
#                    [7, 8], 
#                    [9, 10]])
# t3


# In[ ]:


# 3-dimensional array

t4 = torch.tensor([
    [11,4,2,6],
    [0,5,5,3],
    [2,7,5,4],
    [4,5,6,23],
    [4,1,0,0]
])

print(t4)
print(t4.dtype)

# t4 = torch.tensor([
#     [[11, 12, 13], 
#      [13, 14, 15]], 
#     [[15, 16, 17], 
#      [17, 18, 19.]]])
# t4


# Tensors can have any number of dimensions, and different lengths along each dimension. We can inspect the length along each dimension using the `.shape` property of a tensor.

# In[ ]:


print('t1\n',t1)
print(t1.shape,'\n')
print('t2\n',t2,'\n',t2.shape,'\n')
print('t3\n',t3,'\n',t3.shape,'\n')
print('t4\n',t4,'\n',t4.shape)


# ## Tensor operations and gradients
# 
# We can combine tensors with the usual arithmetic operations. Let's look an example:

# In[ ]:


# Create tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
x, w, b
y = w * x + b
y
# Compute derivatives
y.backward()
# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad) #3 * 4 + 5 = 17


# In[ ]:


# Create tensors.
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)
print(x, w, b)


# In[ ]:


# Arithmetic operations
z = w * x + b
print(z)


# In[ ]:


# Compute derivatives
z.backward()


# In[ ]:


# Display gradients
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad) #3 * 4 + 5 = 17


# As expected, `dy/dw` has the same value as `x` i.e. `3`, and `dy/db` has the value `1`. Note that `x.grad` is `None`, because `x` doesn't have `requires_grad` set to `True`. 
# 
# The "grad" in `w.grad` stands for gradient, which is another term for derivative, used mainly when dealing with matrices. 

# ## Interoperability with Numpy
# 
# [Numpy](http://www.numpy.org/) is a popular open source library used for mathematical and scientific computing in Python. It enables efficient operations on large multi-dimensional arrays, and has a large ecosystem of supporting libraries:
# 
# * [Matplotlib](https://matplotlib.org/) for plotting and visualization
# * [OpenCV](https://opencv.org/) for image and video processing
# * [Pandas](https://pandas.pydata.org/) for file I/O and data analysis
# 
# Instead of reinventing the wheel, PyTorch interoperates really well with Numpy to leverage its existing ecosystem of tools and libraries.

# Here's how we create an array in Numpy:

# In[ ]:


import numpy as np

x = np.array([
    [1,2,3],
    [3,5,6],
    [6,2,4]
 ])
x


# We can convert a Numpy array to a PyTorch tensor using `torch.from_numpy`.

# In[ ]:


# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)
y


# Let's verify that the numpy array and torch tensor have similar data types.

# In[ ]:


x.dtype,y.dtype


# We can convert a PyTorch tensor to a Numpy array using the `.numpy` method of a tensor.

# In[ ]:


# Convert a torch tensor to a numpy array
z = y.numpy()
z


# The interoperability between PyTorch and Numpy is really important because most datasets you'll work with will likely be read and preprocessed as Numpy arrays.

# In[ ]:


get_ipython().system('pip install jovian --upgrade --quiet')
import jovian


# In[ ]:


jovian.commit(project='01-basics-pytorch')


# In[ ]:




