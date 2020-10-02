#!/usr/bin/env python
# coding: utf-8

# # Introduction

# If there are any recommendations/changes you would like to see in this notebook, please leave a comment. Any feedback/constructive criticism would be genuinely appreciated.
# 
# This notebook is always a work in progress. So, please stay tuned for more to come.
# 
# If you like this notebook or find this notebook helpful, Please feel free to **UPVOTE** and/or leave a comment.

# # About Numpy

# NumPy (or Numpy) is a Linear Algebra Library for Python, the reason it is so important for Data Science with Python is that almost all of the libraries in the PyData Ecosystem rely on NumPy as one of their main building blocks.
# 
# Numpy is also incredibly fast, as it has bindings to C libraries. For more info on why you would want to use Arrays instead of lists, check out this great [StackOverflow post](http://stackoverflow.com/questions/993984/why-numpy-instead-of-python-lists).

# # Using Numpy

# In[ ]:


# Once you've installed NumPy you can import it as a library:
import numpy as np


# Numpy has many built-in functions and capabilities. We won't cover them all but instead we will focus on some of the most important aspects of Numpy: vectors,arrays,matrices, and number generation. Let's start by discussing arrays.
# 
# ### Numpy Arrays
# 
# NumPy arrays are the main way we will use Numpy throughout the course. Numpy arrays essentially come in two flavors: vectors and matrices. Vectors are strictly 1-d arrays and matrices are 2-d (but you should note a matrix can still have only one row or one column).
# 
# Let's begin our introduction by exploring how to create NumPy arrays.
# 
# # Creating NumPy Arrays
# 
# ### From a Python List
# 
# We can create an array by directly converting a list or list of lists:

# In[ ]:


my_list = [1,2,3]
my_list


# In[ ]:


np.array(my_list)


# In[ ]:


my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
my_matrix


# In[ ]:


np.array(my_matrix)


# # Built-in Methods
# 
# There are lots of built-in ways to generate Arrays

# ### arange
# 
# Return evenly spaced values within a given interval.

# In[ ]:


np.arange(0,10)


# In[ ]:


np.arange(0,11,2)


# ### zeros and ones
# 
# Generate arrays of zeros or ones

# In[ ]:


np.zeros(3)


# In[ ]:


np.zeros((5,5))


# In[ ]:


np.ones(3)


# In[ ]:


np.ones((3,3))


# ### linspace
# Return evenly spaced numbers over a specified interval.

# In[ ]:


np.linspace(0,10,3)


# In[ ]:


np.linspace(0,10,50)


# ### eye
# 
# Creates an identity matrix

# In[ ]:


np.eye(4)


# ### Random 
# 
# Numpy also has lots of ways to create random number arrays:
# 
# ### rand
# Create an array of the given shape and populate it with
# random samples from a uniform distribution
# over ``[0, 1]``.

# In[ ]:


np.random.rand(2)


# In[ ]:


np.random.rand(5,5)


# ### randn
# 
# Return a sample (or samples) from the "standard normal" distribution. Unlike rand which is uniform:

# In[ ]:


np.random.randn(2)


# In[ ]:


np.random.randn(5,5)


# ### randint
# Return random integers from `low` (inclusive) to `high` (exclusive).

# In[ ]:


np.random.randint(1,100)


# In[ ]:


np.random.randint(1,100,10)


# # Array Attributes and Methods
# 
# Let's discuss some useful attributes and methods or an array:

# In[ ]:


arr = np.arange(25)
ranarr = np.random.randint(0,50,10)


# In[ ]:


arr


# In[ ]:


ranarr


# # Reshape
# Returns an array containing the same data with a new shape.

# In[ ]:


arr.reshape(5,5)


# ### max,min,argmax,argmin
# 
# These are useful methods for finding max or min values. Or to find their index locations using argmin or argmax

# In[ ]:


ranarr


# In[ ]:


ranarr.max()


# In[ ]:


ranarr.argmax()


# In[ ]:


ranarr.min()


# In[ ]:


ranarr.argmin()


# # Shape
# 
# Shape is an attribute that arrays have (not a method):

# In[ ]:


# Vector
arr.shape


# In[ ]:


# Notice the two sets of brackets
arr.reshape(1,25)


# In[ ]:


arr.reshape(1,25).shape


# In[ ]:


arr.reshape(25,1)


# In[ ]:


arr.reshape(25,1).shape


# ### dtype
# 
# You can also grab the data type of the object in the array:

# In[ ]:


arr.dtype


# # Numpy Indexing and Selection
# In this lecture we will discuss how to select elements or groups of elements from an array.

# In[ ]:


import numpy as np


# In[ ]:


#Creating sample array
arr = np.arange(0,11)


# In[ ]:


#Show
arr


# # Bracket Indexing and Selection
# The simplest way to pick one or some elements of an array looks very similar to python lists:

# In[ ]:


#Get a value at an index
arr[8]


# In[ ]:


#Get values in a range
arr[1:5]


# In[ ]:


#Get values in a range
arr[0:5]


# # Broadcasting
# 
# Numpy arrays differ from a normal Python list because of their ability to broadcast:

# In[ ]:


#Setting a value with index range (Broadcasting)
arr[0:5]=100

#Show
arr


# In[ ]:


# Reset array, we'll see why I had to reset in  a moment
arr = np.arange(0,11)

#Show
arr


# In[ ]:


#Important notes on Slices
slice_of_arr = arr[0:6]

#Show slice
slice_of_arr


# In[ ]:


#Change Slice
slice_of_arr[:]=99

#Show Slice again
slice_of_arr


# In[ ]:


#Now note the changes also occur in our original array!


# In[ ]:


arr


# In[ ]:


#Data is not copied, it's a view of the original array! This avoids memory problems!


# In[ ]:


#To get a copy, need to be explicit
arr_copy = arr.copy()

arr_copy


# # Indexing a 2D array (matrices)
# 
# The general format is **arr_2d[row][col]** or **arr_2d[row,col]**. I recommend usually using the comma notation for clarity.

# In[ ]:


arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))

#Show
arr_2d


# In[ ]:


#Indexing row
arr_2d[1]


# In[ ]:


# Format is arr_2d[row][col] or arr_2d[row,col]

# Getting individual element value
arr_2d[1][0]


# In[ ]:


# Getting individual element value
arr_2d[1,0]


# In[ ]:


# 2D array slicing

#Shape (2,2) from top right corner
arr_2d[:2,1:]


# In[ ]:


#Shape bottom row
arr_2d[2]


# In[ ]:


#Shape bottom row
arr_2d[2,:]


# # Fancy Indexing
# 
# Fancy indexing allows you to select entire rows or columns out of order,to show this, let's quickly build out a numpy array:

# In[ ]:


#Set up matrix
arr2d = np.zeros((10,10))


# In[ ]:


#Length of array
arr_length = arr2d.shape[1]


# In[ ]:


#Set up array

for i in range(arr_length):
    arr2d[i] = i
    
arr2d


# In[ ]:


#Fancy indexing allows the following


# In[ ]:


arr2d[[2,4,6,8]]


# In[ ]:


#Allows in any order
arr2d[[6,4,2,7]]


# # Selection
# 
# Let's briefly go over how to use brackets for selection based off of comparison operators.

# In[ ]:


arr = np.arange(1,11)
arr


# In[ ]:


arr > 4


# In[ ]:


bool_arr = arr>4


# In[ ]:


bool_arr


# In[ ]:


arr[bool_arr]


# In[ ]:


arr[arr>2]


# In[ ]:


x = 2
arr[arr>x]


# # NumPy Operations

# ## Arithmetic
# You can easily perform array with array arithmetic, or scalar with array arithmetic. Let's see some examples:

# In[ ]:


import numpy as np
arr = np.arange(0,10)


# In[ ]:


arr + arr


# In[ ]:


arr * arr


# In[ ]:


arr - arr


# In[ ]:


# Warning on division by zero, but not an error!
# Just replaced with nan
arr/arr


# In[ ]:


# Also warning, but not an error instead infinity
1/arr


# In[ ]:


arr**3


# ## Universal Array Functions
# 
# Numpy comes with many [universal array functions](http://docs.scipy.org/doc/numpy/reference/ufuncs.html), which are essentially just mathematical operations you can use to perform the operation across the array. Let's show some common ones:

# In[ ]:


#Taking Square Roots
np.sqrt(arr)


# In[ ]:


#Calcualting exponential (e^)
np.exp(arr)


# In[ ]:


np.max(arr) #same as arr.max()


# In[ ]:


np.sin(arr)


# In[ ]:


np.log(arr)


#  # Thanks !!
# That's all we need to know for now!
