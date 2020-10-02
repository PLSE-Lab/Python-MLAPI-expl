#!/usr/bin/env python
# coding: utf-8

# ## **NumPy** tutorials
# 
# Items covered in the course  
# * Understanding the basic functionalities
# * Understanding Array Indexing
# * Datatypes
# * Array Understanding
# * Broadcasting
# * Image Operations
# 
# I would like to thank [Standford CS class](http://cs231n.github.io/) for this beautifully designed course.
# 
# Let us begin!!!
# 
# # Numpy
# >[Numpy]() is a fundamental package for scientific computing. It can be used to perform following operations:
# * A powerful N-dimesional Array object
# * sophasticated broadcasting functions
# * Capabilities to use linear algebra, fourier transformation and random number generation

# In[ ]:


#import for numpy

import numpy as np


# ## Basic functionalities

# * Working with arrays and shape

# In[ ]:


list = [1,2,3] # Array of rank one
numpyList = np.array(list)
print("NumpyList ",numpyList)
print(type(numpyList))
print("Elements of list ", numpyList[0],numpyList[1],numpyList[2])
print("Shape of numpy list ",numpyList.shape)


# #### Numpy array manipulations

# In[ ]:


numpyList[0] = 5
print("The List after replacing 1 with 5 is ", numpyList)


# In[ ]:


b = np.array([[12,13,14],[13,14,15]]) #Arry of rank-2


# In[ ]:


print("List elements are \n", b)
print("Shape of the numpy array ", b.shape)


# # Functionalities of NumPy
# * Functionalities offered in numpy. These functions can be used for matrix operations and linear algebra
#  * Zeros  
#  _np.zeros(shape)_
#  * Ones
#  _np.ones(shape)_
#  * Full  
#  Creates a constant array of shape  
#  _np.full(shape)_
#  * Eye  
#  Creates an identity matrix with diagonals being 1  
#  _np.eye(N)_
#  * Random (He is amazing)
#      * _np.random.random(size=)_
#      * _np.random.randint(shape)_
#  

# In[ ]:


#Zeroes
np.zeros(shape=(2,2))


# In[ ]:


## Ones
np.ones(shape=(3,3))


# In[ ]:


## Full
np.full(shape=(4,4),fill_value=10)


# In[ ]:


### Eye
np.eye(3) # giving N=3 provides a 3X3 identity matrix


# In[ ]:


## Random matrix creation using numpy

np.random.random(size=(3,3))


# In[ ]:


# Fix a seed and create a numpy matrix
np.random.seed(101)
np.random.randint(low=0,high=50,size=(10,5))


# # Array Indexing
# **Slicing**, a numpy array can be sliced for a sub array and it can be manipulated.
# 
# **Note: One should know slicing array and changing it will change the original array, so it is better to make a copy**

# In[ ]:


np.random.seed(123)
array = np.random.randint(low=1,high=50,size=(10,5))


# In[ ]:


array


# In[ ]:


copyOfArray = array.copy()


# In[ ]:


slice = copyOfArray[:2,1:3] # Take first to rows and fetch 1 and 2 columns of the matrix


# In[ ]:


slice


# In[ ]:


copyOfArray[:,1:2]


# In[ ]:


#Original array changed
array[:1,:] = 9999
array


# In[ ]:


copyOfArray[3:6,2:5]


# In[ ]:


# Note: Array Indexing will yield a lower rank array


# In[ ]:


copyOfArray[1,:].shape # Rank-1 array


# In[ ]:


copyOfArray[1:2,:].shape # Rank-2 array


# 1. ### Boolean array indexing
# There is always chance to perform conditional based selection for a matrix. A long matrix could be **compared** (math based skills could be applied to select a subset of matrix to matching condition).

# In[ ]:


np.random.seed(3213)
a = np.random.randint(low=1,high=50,size=(5,10))


# In[ ]:


a[a>20].reshape((6,5))


# ## Datatypes  
# Numpy tries to guess the datatype for the array. There are also functionalities supported to construct array of different type  
# Some of the examples are illustrated below:

# In[ ]:


print(np.array([1,2,3]).dtype)


# In[ ]:


print(np.array([1.0,2,3.8]).dtype)


# In[ ]:


a=np.array([[1,2,3],[3,4,5]],dtype=np.float64)
print(a.dtype)


# ## Arithmetic Operations
# Some of the arithmetic operations can be done using numpy  
# We will see this with some examples:  

# In[ ]:


np.random.randint(312)
a = np.random.randint(low=0,high=10,size=(3,3))
b = np.random.randint(low=11,high=20,size=(3,3))
print("Matrix A")
print(a)
print()
print("Matrix B")
print(b)


# In[ ]:


# Addition of matrix
np.add(a,b)


# In[ ]:


# Sub of matrix
np.subtract(a,b)


# In[ ]:


# Multiply of matrix
np.multiply(a,b)


# In[ ]:


# Divide of matrix
np.divide(a,b)


# In[ ]:


# SQRT of matrix
np.sqrt(a)


# ### One of the important aspects of matrix are _dot product_ and _sum of element over (rows and columns)_
# * Dot Product  
# Dot product in numpy is an element wise vector multiplication. In numpy it can be achieved both by _numpy.dot(a,b)_ or _as a.dot(b)_ ... _where a,b are matrices_  
# * Sum of elements  
# Sum of each elements is compute either via columns or rows using numpy. It can be done using _numpy.sum(a)_ or _numpy.sum(a,axis=1|0)_ ... _where a,b are matrices_

# In[ ]:


a


# In[ ]:


np.dot(a,b)


# In[ ]:


a.dot(b)


# In[ ]:


np.sum(a)


# In[ ]:


np.sum(a,axis=0) ## Accross Column sum


# In[ ]:


np.sum(a,axis=1) ## Accross rows sum


# I hope this tutorials provide some basic understanding with numpy. I would like to take this to advanced in my next part where i would try to solve some equations using numpy  

# In[ ]:




