#!/usr/bin/env python
# coding: utf-8

# ## NumPy Basics Tutorial ##
# by Briane Paul Samson
# 
# Welcome to our first tutorial in the COMET Data Science Workshops. This notebook will walk you through some basic NumPy functions and operations that you can use in any data analysis and data science project.
# 
# 
# ----------
# 
# 
# From [numpy.org][1]:
# 
# 
# > *NumPy is the fundamental package for scientific computing with Python. It contains among other things: a powerful N-dimensional array
# > object sophisticated (broadcasting) functions tools for integrating
# > C/C++ and Fortran code useful linear algebra, Fourier transform, and
# > random number capabilities Besides its obvious scientific uses, NumPy
# > can also be used as an efficient multi-dimensional container of
# > generic data. Arbitrary data-types can be defined. This allows NumPy
# > to seamlessly and speedily integrate with a wide variety of
# > databases.*
# 
# Source: http://nbviewer.jupyter.org/github/twistedhardware/mltutorial/blob/master/notebooks/IPython-Tutorial/4%20-%20Numpy%20Basics.ipynb
# 
#   [1]: http://www.numpy.org/

# ## Importing NumPy ##

# In[ ]:


import numpy as np


# ## Creating Arrays ##
# NumPy allows you to create and operate on homogeneous multidimensional arrays faster than in standard Python.
# 
# **ndarray** (or its alias *array*) is NumPy's array class. A dimension is called an *axis*. The number of axes is called *rank*.
# 
# In creating an array from scratch, we can use **np.arange** with the following syntax:
# 
# **np.arange([start,] stop[, step,], dtype=None)**

# In[ ]:


np.arange(10)


# In[ ]:


np.arange(1, 10)


# In[ ]:


np.arange(1, 10, 2)


# In[ ]:


np.arange(1, 20, 3, dtype=np.float64)


# Or use **array** for Python lists and sequences.
# 

# In[ ]:


x = np.array([1, 3, 5, 7])
x


# In[ ]:


type(x)


# In[ ]:


np.array([(1, 3, 5), (7, 9, 11), (13, 15, 17)]).ndim


# In[ ]:


np.zeros((2, 3), dtype=np.int16)


# In[ ]:


np.ones((3, 3), dtype=np.int16)


# In[ ]:


np.empty((2, 2, 4))


# In[ ]:


np.linspace(2, 4, 10) #better for floating point arguments because of precision


# In[ ]:


np.random.random((3, 3))


# Important Array Attributes
# --------------------
# 
# **Dimensions**
# 
# the number of axes (dimensions) of the array. In the Python world, the number of dimensions is referred to as rank.

# In[ ]:


ds = np.arange(1, 10, 3)
print(ds)
ds.ndim


# In[ ]:


threeD = np.arange(1, 30, 2).reshape(3, 5)
print(threeD)


# **Shape**
# 
# a tuple indicating the size of the array in each dimension. For a matrix with *n* rows and *m* columns, shape will be **(n, m)**
# 

# In[ ]:


ds.shape


# **Size**
# 
# the total number of elements in the array. This is the same as the product of *n* and *m* in **shape**.
# 

# In[ ]:


ds.size


# **Data Type**
# 
# the data type of the elements.
# 

# In[ ]:


ds.dtype


# **Item Size**
# 
# the size in bytes of the elements (i.e. `float64 = 64/8 = 8`).
# 
# 

# In[ ]:


ds.itemsize


# Basic Operations
# ----------------
# Arithmetic operators on arrays apply elementwise. A new array is created and filled with the result.

# In[ ]:


a = np.arange(5)
b = np.array([2, 4, 0, 1, 2])
a


# In[ ]:


diff = a-b
diff


# In[ ]:


b**2


# In[ ]:


2*b


# In[ ]:


np.sin(a)


# In[ ]:


np.sum(a)


# In[ ]:


np.max(a)


# In[ ]:


np.min(a)


# In[ ]:


b > 2


# In[ ]:


a*b #by default, matrix multiplication is elementwise


# In[ ]:


x = np.array([[1,1], [0,1]])
y = np.array([[2,0], [3,4]])
x*y


# In[ ]:


x.dot(y) #same as np.dot(x, y)


# In[ ]:


x.sum()


# In[ ]:


x.sum(axis=0)


# In[ ]:


x.sum(axis=1)


# In[ ]:


z = np.random.random((3, 4))
z


# In[ ]:


np.mean(z)


# In[ ]:


np.median(z)


# In[ ]:


np.std(z)


# Reshaping Arrays
# ----------------

# In[ ]:


data_set = np.random.random((2,3))
data_set


# In[ ]:


np.reshape(data_set, (3,2))


# In[ ]:


np.reshape(data_set, (6,1))


# In[ ]:


np.reshape(data_set, (6))


# In[ ]:


np.ravel(data_set)


# Slicing Arrays
# --------------

# In[ ]:


data_set = np.random.random((5,10))
data_set


# In[ ]:


data_set[1]


# In[ ]:


data_set[1][0]


# In[ ]:


data_set[1,0]


# **Slicing range of data**
# 

# In[ ]:


data_set[2:4]


# In[ ]:


data_set[2:4,0]


# In[ ]:


data_set[2:4,0:2]


# In[ ]:


data_set[:,0]


# **Slicing data with steps**
# 

# In[ ]:


data_set[2:5:2]


# In[ ]:


data_set[::]


# In[ ]:


data_set[::2]


# In[ ]:


data_set[2:4]


# In[ ]:


data_set[2:4,::2]


# In[ ]:




