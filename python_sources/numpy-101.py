#!/usr/bin/env python
# coding: utf-8

# 
# 
# Numpy 101
# 
# This notebook represents the first step of my learning journey about Numpy. I included my learning journey resources in the references section.
# 
# What is Numpy?
# 
# NumPy is the fundamental package for scientific computing in Python. NumPy is a library for the Python programming language, adding support for large, 
# multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
# 
# *These exercises are  Quickstart tutorial on numpy.org.*
# 

# In[ ]:


import numpy as np


# In[ ]:


a = np.arange(20).reshape(4,5)
a


# In[ ]:


a.shape


# In[ ]:


a.ndim


# In[ ]:


a.dtype.name


# In[ ]:


a.itemsize


# In[ ]:


a.size


# In[ ]:


type(a)


# In[ ]:


c = np.array( [ [1,2], [3,4] ], dtype=complex )
c


# In[ ]:


np.empty( (2,3) )


# In[ ]:


a = np.array( [20,30,40,50] )
a


# In[ ]:


b = np.arange( 4 )
b


# In[ ]:


c = a-b
c


# In[ ]:



b=b**2
b


# In[ ]:


10*np.sin(a)


# In[ ]:


a<35


# In[ ]:


A = np.array( [[1,1],
               [0,1]] )
   
B = np.array( [[2,0],
               [3,4]] )


# In[ ]:


A * B


# In[ ]:


A @ B 


# In[ ]:


A.dot(B) 


# In[ ]:


from numpy import random 
from numpy.random import default_rng
rg = default_rng() #random generation
a = np.ones((2,3), dtype=int)
b = rg.random((2,3))
print(a)
b


# In[ ]:


a *= 3
a


# In[ ]:


b += a
b


# In[ ]:


from math import pi

a = np.ones(3, dtype=np.int32)
b = np.linspace(0,pi,3)
b.dtype.name


# In[ ]:


c = a+b
c


# In[ ]:


c.dtype.name


# In[ ]:


d = np.exp(c*1j)
print(d)
d.dtype.name


# In[ ]:


a = rg.random((2,3))
a


# In[ ]:


print(a.sum())
print(a.min())
print(a.max())


# In[ ]:


b = np.arange(12).reshape(3,4)
b


# In[ ]:


b.sum(axis=0) # sum of each column


# In[ ]:


b.min(axis=1) # min of each row


# In[ ]:


b.cumsum(axis=1) # cumulative sum along each row


# In[ ]:


a = np.arange(10)**3
a


# In[ ]:


a[2]


# In[ ]:


a[2:5]


# In[ ]:


a[:6:2] = 1000 # from start to position 6, exclusive, set every 2nd element to 1000
a


# In[ ]:


a[ : :-1] 
a


# In[ ]:


a = np.floor(10*rg.random((3,4)))

print(a.shape)
a


# In[ ]:


a.ravel()  # returns the array, flattened


# In[ ]:


a.reshape(6,2)  # returns the array with a modified shape


# In[ ]:


a.T  # returns the array, transposed


# In[ ]:


a.T.shape


# In[ ]:


a = np.floor(10*rg.random((2,2)))
b = np.floor(10*rg.random((2,2)))
print(a)
print(b)


# In[ ]:


np.vstack((a,b))


# In[ ]:


np.hstack((a,b))


# In[ ]:


from numpy import newaxis
np.column_stack((a,b)) 


# In[ ]:


a = np.array([4.,2.])
b = np.array([3.,8.])
np.column_stack((a,b))


# In[ ]:


np.hstack((a,b))  


# In[ ]:


a[:,newaxis] 


# In[ ]:


np.column_stack((a[:,newaxis],b[:,newaxis]))


# In[ ]:


np.hstack((a[:,newaxis],b[:,newaxis])) 


# In[ ]:


np.column_stack is np.hstack


# In[ ]:


np.row_stack is np.vstack


# In[ ]:


a = np.array((1,2,3))

b = np.array((2,3,4))


# In[ ]:


np.column_stack((a,b))


# In[ ]:


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
np.concatenate((a, b), axis=0)


# In[ ]:


np.concatenate((a, b.T), axis=1)


# In[ ]:


np.concatenate((a, b), axis=None)


# In[ ]:


a = np.arange(12)
b = a   
b is a


# In[ ]:


a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)


# In[ ]:


a.transpose()


# In[ ]:


np.linalg.inv(a)


# In[ ]:




