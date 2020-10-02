#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import os


# * NumPy ( or numpy ) is a Linear Algebra Library for Python.
#   Almost all the libraries in the PyData system rely on Numpy
# * It is also fast as it has binding to C libraries.

#  * Numpy  arrays come in two flavors -- > Vectors or Matrices
# >  * Vectors are strictly 1- d arrays , however Matrix can still have one or more than one columns or rows

# ### Numpy Arrays

# In[ ]:


mylis = a = [1, 2, 3]
mylis


# In[ ]:


np.array(mylis)


# In[ ]:


np_arr = np.array(mylis)
np_arr


# In[ ]:


# A 2-d array

mat = [[1,2,3],[4,5,6],[7,8,9]]  # Two sets of brackets 
# indicating 2d

np_mat = np.array(mat)
np_mat


# In[ ]:


# press Shift + tab for quick documentation.
#All even numbers between 0 and 10
np.arange(0,11,2) #Similar to python's built in range function


# In[ ]:


np.zeros(3)


# In[ ]:


np.zeros((2,3))  # 2 rows, 3 columns


# In[ ]:


np.ones(4)


# In[ ]:


np.ones ((6,2))


# In[ ]:


# Linspace returns evenly spaced numbers over an interval

np.linspace(0,11,num = 3)
# 0 , 5.5 , 11 (Note unlike .arange() 11 is included)


# In[ ]:


# Identity matrix
np.eye(4)  # Single digit input because Identity matrices are square matrics


# In[ ]:


#Creates an array and populates it with uniformly distributed data.
np.random.rand(5) #1-D array


# In[ ]:


np.random.rand(5,5) # 5x5 matrix of random (uniformly distributed numbers)


# In[ ]:


# If you need numbers from normal distribution
np.random.randn(2) # 


# In[ ]:


np.random.rand(4,4)


# In[ ]:


np.random.randint(1, 100)
# gives an randome integer from 1 and 100
# 1 is inclusive and 100 is exclusive


# In[ ]:


np.random.randint(1, 100, 10)


# In[ ]:


arr = np.arange(25)
arr


# In[ ]:


ranarr = np.random.randint(0,50,10)
ranarr


# In[ ]:


# reshape method 


# In[ ]:


arr.reshape(5,5) # pass in the new dimensions you want
# 5 * 5 = 25 elements therefore no error


# In[ ]:


ranarr


# In[ ]:


ranarr.max()


# In[ ]:


ranarr.min()


# In[ ]:


# Location of max  value.

ranarr.argmax()


# In[ ]:


# Location of min value.
ranarr.argmin()


# In[ ]:


arr.shape  # 25 columns and 1 row


# In[ ]:


arr = arr.reshape(5,5)


# In[ ]:


arr.shape  # shape is an attribute


# In[ ]:


# Data in 64bit integers!
arr.dtype # dtype is also an attribute


# ## Numpy indexing and Selection

# In[ ]:


arr = np.arange(0,11)


# In[ ]:


arr


# In[ ]:


arr[2:]


# In[ ]:


arr[::-1] # reverse the array!
 


# In[ ]:


arr[1 : 5] 
# start at 1 and go upto 5 (5 not included )! that means 4!


# In[ ]:


# Broadcasting
arr[:5] = 100

arr


# In[ ]:


arr = np.arange(0,11)


# In[ ]:


slice_arr = arr[:6]


# In[ ]:


slice_arr


# In[ ]:


slice_arr[:] = 99


# In[ ]:


arr  # This means that slice_arr  is only a Referance to the  array object, just as in python lists.


# In[ ]:


# If you want a copy
arr_copy = arr.copy()


# In[ ]:


arr_copy[:] = 100
arr_copy


# In[ ]:


arr # Now its not getting changed (deep copy)


# In[ ]:


b = arr[:] # Even this does not create deep copy  phew!
b[:] = 3213


# In[ ]:


b


# In[ ]:


arr #Even this does not create deep copy  phew!


# ### Numpy indexing and selection

# In[ ]:


arr_2d = np.arange(5,46,5).reshape(3,3)


# In[ ]:


arr_2d


# In[ ]:


arr_2d[0][0]


# In[ ]:


arr_2d[0]


# In[ ]:


arr_2d[:2][:1]


# In[ ]:


arr_2d[1,2] # 1st row 2nd column


# In[ ]:


# Grab submatrices

arr_2d


# In[ ]:


arr_2d[:2,1:]

#grab everything from row 0 and row 1 where column ranges from 1 onwards


# In[ ]:


arr_2d[1:,:2][::-1]  # Mirror image of the subsection


# In[ ]:


arr = np.arange(1,11)


# In[ ]:


arr > 5 # gives and array of boolean values


# In[ ]:


b = arr > 5


# In[ ]:


# Use it to do conditional selection


# In[ ]:


arr[b]  # returns where b is true


# In[ ]:


# In one step
arr[arr>5]


# In[ ]:


# All the elements of array less than three!


# In[ ]:


arr[arr<=3]


# In[ ]:


arr[arr <=3]


# In[ ]:


arr_2d = np.arange(50).reshape(5,10)
arr_2d


# In[ ]:


arr_2d[1:3,3:5]


# In[ ]:


arr_2d[:,3] # Only the 3rd column!!!


# ## Numpy operations

# In[ ]:


arr = np.arange(0,11)
arr


# In[ ]:


arr + 1

# Add 1 to every element of the array


# In[ ]:


arr - 1


# In[ ]:


arr * arr
# Looks like a square !


# In[ ]:


arr + 100


# In[ ]:


arr / 2


# In[ ]:


arr / arr 
# Its giving a runtime warning!!
# See that 0/0 in the output is given as a NaN aka Null


# In[ ]:


1 / arr

# Now you are getting inf ( meaning infinity)


# In[ ]:


arr ** 2

# Squared


# In[ ]:


np.sqrt(arr)
# square root of every element in the array


# In[ ]:


np.max(arr)


# In[ ]:


np.min(arr)


# In[ ]:


np.sin(arr)


# In[ ]:


np.log(arr)

# Gives -infinty value!! no error 
# Only runtime error


# In[ ]:




