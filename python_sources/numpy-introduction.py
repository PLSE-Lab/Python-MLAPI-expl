#!/usr/bin/env python
# coding: utf-8

# # Introduction to NumPy

# 26/07/2019 - Machine Learning Lab 2

# ## Imports

# In[ ]:


import numpy as np
import os


# ## What is NumPy

# NumPy is a library used to extend the numerical computational capacity of vanilla Python. It offers a powerful N-dimensional array object that can be manipulated and mutated effeciently. It also offers a collection of high-level mathematical functions to operate on these arrays.

# In[ ]:


# Run this to see the documentation of NumPy
get_ipython().run_line_magic('pinfo', 'np')


# ## Basic functions

# We can take a look at the documentation to see some of the functions NumPy provides. Alternatively, you can type `numpy.` + `tab` to see what the autocomplete suggests.

# `np.array` generates an NumPy array and returns it. It supports created an N-dimensional array from initialization. The function accepts a python list as an argument.

# In[ ]:


get_ipython().run_line_magic('pinfo', 'np.array')


# In[ ]:


arr = np.array([5, 10, 15, 20])

print(arr)
print(np.flip(arr))


# If we need to generate an NumPy array without a pre-existing list, we can use the `np.arrange` method to create one. Both methods create the same type of object and therefore have all of the same methods.

# In[ ]:


new_arr = np.arange(15)

print(new_arr)

print(type(arr))
print(type(new_arr))


# ## Questions

# 1. Write a program of a two dimensional array using NumPy and print elements, reverse the two of the 2D array and print the elements.

# In[ ]:


arr = np.array([
    [1, 2, 3], 
    [4, 5, 6],
    [7, 8, 9]
])

np.flip(arr)


# 2. Generate the digits upto 15 using `np.arange` method

# In[ ]:


arr = np.arange(1, 16)
print(arr)


# 3. Print the common items from and two NumPy arrays also print the position of the common elements and remove the common items and print the different elements also of two NumPy arrays

# In[ ]:


array_one = np.arange(1, 15)
array_two = np.arange(5, 20)

for elem in array_one:
    if elem in array_two:
        print(elem, end=' ')

print('\n')
for elem in array_one:
    if elem not in array_two:
        print(elem, end=' ')


# 4. Write a NumPy program to generate the 10 random integers between 10 and 100

# In[ ]:


array = np.random.randint(low=10, high=100, size=10)

print(array)


# 5. Generate a random 4x4 matrix of integers ranging from 10 to 100 using NumPy and also to use m the nd minimum values

# In[ ]:


array = np.random.random((4, 4))
print('Original array ->', array)
xmin, xmax = array.min(), array.max()
print("Minimum and maximum values ->", xmin, xmax)

