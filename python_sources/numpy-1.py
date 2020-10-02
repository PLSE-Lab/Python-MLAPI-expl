#!/usr/bin/env python
# coding: utf-8

# #                                                    NUMPY-1

# Python for Data Science- Module1

# What Numpy can do-
# - Mathematical and logical operations on Arrays
# - Fourier transform
# - Linear Algebra Operation 
# - Random Number Generation
# 

# ### Creating Array

# Syntax: numpy.array(object)

# In[ ]:


import numpy as np
x=np.array([1,4,6,3])
print(type(x))# return the type of array i.e numpy.ndarray, nd means n-dimensional
print(x)


# In[ ]:


x=np.array([1,2,5,'v','q',5,6])
print(type(x))
print(x)


# Since here all data inside the array got converted into string because we enter mixed data in the array, so by default the numpy converts each element of the array to same data type.

# ### Generating Array- Using np.linspace()

# Syntax: np.linspace(start,stop,num,dtype,endpoint=True/False,retstep=True/False)
# - num: total number of Samples you want to generate
# - retstep: return Step/ incremented value/ value by which your data samples got incremented
# - endpoint: if True then includes the stop value in the list

# In[ ]:


#here stop value 9 is not included because endpoint=False 
x=np.linspace(start=3,stop=9,endpoint=False,retstep=False)
print(x)


# In[ ]:


#here stop value 9 is included because endpoint=True and increment value is also returned as: 0.12244897959183673
x=np.linspace(start=3,stop=9,endpoint=True,retstep=True)
print(x)


# ### Generating Array- Using np.arange()

# Synatx: np.arange(start,stop,step)
# - step: increment value

# In[ ]:


x=np.arange(5,25,5)
print(x)#x will never includes stop values


# ### Generating Array- Using np.ones()/zeros() 

# Synatx:  
# - np.zeros(shape,dtype)
# - np.ones(shape,dtype)
# - shape: (row,column)

# In[ ]:


x=np.zeros((2,3))
print(x)


# In[ ]:


x=np.ones((5,3),int)
print(x)


# ### Generating Random Array- random.rand()

# Syntax: np.random.rand(shape)
#     

# In[ ]:


x=np.random.rand(2,4)
print(x)


# ### Generating Random Array- np.logspace()

# Returns equally spaced number based on log values
# 
# Syntax: np.logspace(start,stop,num,endpoint,base,dtype)
# - base: base value of log      (default: 10.0)
# - num: number of samples       (default: 50)

# In[ ]:


x=np.logspace(4,50,10,endpoint=True,base=4.0,dtype=int)
print(x)


# ### Calculating Time using timeit module 

# In[ ]:


import timeit #determines the time of program
x=range(1000)
get_ipython().run_line_magic('time', 'sum(x)')


# In[ ]:


x=range(1000)
get_ipython().run_line_magic('timeit', 'sum(x)')


# ### Storage Space Numpy

# getsizeof(): Return Size of in bytes
# - Synatx: sys.getsizeof(object)
# 
# itemsize(): Return Size of one element of numpy array
# - Syntax: numpy.ndarray.itemsize

# In[ ]:


import sys
x=np.arange(1,1000)
a=sys.getsizeof(1)#returns size of an element in bytes
b=sys.getsizeof(1)*len(x)# returns whole size of list/array
print('a:',str(a) + ' bytes')
print('b:',str(b) + ' bytes')


# In[ ]:


x=np.arange(1,1000)
x.itemsize


# In[ ]:


x.itemsize*x.size

