#!/usr/bin/env python
# coding: utf-8

# ![Practice Numpy ](http://https://en.wikipedia.org/wiki/NumPy#/media/File:NumPy_logo.svg)

# In[ ]:


#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.
#Along with a large collection of high-level mathematical functions to operate on these arrays.


# # Get the numpy version and show numpy build configuration

# In[ ]:


import numpy as np 
print(np.__version__)
print(np.show_config)


# # Converting List/list of lists into numpy array

# In[ ]:


my_list = [1,2,3,4]
np.array(my_list)


# In[ ]:


list_of_list = [['a','b','c'],[1,2,3]]
np.array(list_of_list)


# # Converting Dict into numpy array

# In[ ]:



my_dict = {'a':1,'b':2,'c':3,'d':4}
np.array(my_dict)


# # Converting multiple data type into numpy array

# In[ ]:


data = [[1,2,3,4,5],('a','b','c','d'),{'a':1,'b':2,'c':3,'d':4},[7,8,9,10]]
np.array(data)


# # Matrix 

# In[ ]:


matrix = [[1,2,3],[4,5,6],[7,8,9]]
matrix


# In[ ]:


np.array(matrix)


# # Build in Method
#  
# 2. add    - add : ndarray or scalar The sum of `x1` and `x2`, element-wise.  Returns a scalar if both  `x1` and `x2` are scalars
# 3.

# # arange : return evenly spaced value with in s given interval
# 

# In[ ]:


np.arange(7)


# In[ ]:


#generate array from '0'(include) to '10' (exclude)
np.arange(0,10)


# In[ ]:


#generate array from '0'(include) to '10' (exclude) with step size '2' and datatype =float
np.arange(start= 0,stop = 10,step= 2, dtype=float)


# # add : ndarray or scalar , The sum of `x1` and `x2`, element-wise.Returns a scalar if both  `x1` and `x2` are scalars

# In[ ]:


np.add(5,7)


# shapes should be same then it will add

# In[ ]:


L1 = np.arange(1,10,2)
L2 = np.arange(0,10)
np.add(L1,L2)


# In[ ]:


L1 = np.arange(1,20,2) # odd number
print(L1)
L2 = np.arange(0,10)
print(L2)
np.add(L1,L2)


# adding two matrixs

# In[ ]:


matrix_1 =np.arange(9).reshape(3,3)
print(matrix_1)
print('---------------------')
matrix_2 = np.arange(3)
print(matrix_2)
print("--------------------")
np.add(matrix_1,matrix_2)


# shape will give the dimemsion of the array

# In[ ]:


print(matrix_1.shape)
print(matrix_2.shape)


# # linspace : Return evenly spaced numbers over a specified interval.

# In[ ]:


np.linspace(start = 0,stop = 3)  #bydefault no. of Sample size = 50 


# In[ ]:


np.linspace(start = 2 , stop = 3 , num =5)


# In[ ]:


np.linspace(7,9,5,retstep=True)


# # Zero : Return a new array of given shape and type, filled with zeros

# In[ ]:


np.zeros(shape=5)


# In[ ]:


np.zeros(shape =8 ,dtype = int)


# In[ ]:


np.zeros(shape=(2,2))


# # Ones : Return a new array of given shape and type, filled with ones

# In[ ]:


np.ones(7)


# In[ ]:


np.ones(7,dtype=int)


# In[ ]:


np.ones(shape=(2,2))


# # Eye : Return a 2-D array with ones on the diagonal and zeros elsewhere

# In[ ]:


np.eye(6,dtype=int)


# In[ ]:


np.eye(6,k=1)


# In[ ]:


np.eye(6,k=-1)


# # Random - Numpy also has lots of ways to create random arrays

# # rand

# In[ ]:


#To Create an array of the given shape & populate , it with random sample from a "uniform distribution" over (0,1)
np.random.rand(2)


# In[ ]:


np.random.rand(4,4)


# # randn

# In[ ]:


#To Create an array of the given shape & populate , it with random sample from a  "Standard Normal "distribution.
print(np.random.randn())
print('-------------------------------------------')
print(np.random.randn(5))


# In[ ]:


np.random.randn(3,3)


# # randint

# In[ ]:


#To Create an array of the given shape & populate , it with random sample from a  "Discrete uniform " distribution.
#retun random integer from low(inclusive) to high (exclusive)

print(np.random.randint(10))
print("---------------------------------------")
print(np.random.randint(low = 2,high = 20 ,size =10 ))
print('------------------------------------------------')


# In[ ]:


print(np.random.randint(low = 2,high = 20 ,size =10 ,dtype = float))


# In[ ]:


np.random.randint(5,size = (3,3))

