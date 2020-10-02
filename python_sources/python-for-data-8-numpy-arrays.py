#!/usr/bin/env python
# coding: utf-8

# # Python for Data 8: Numpy Arrays
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)

# Python's built in data structures are great for general-purpose programming, but they lack some specialized features we'd like for data analysis. For example, adding rows or columns of data in an element-wise fashion and performing math operations on two dimensional tables (matrices) are common tasks that aren't readily available with Python's base data types. In this lesson we'll learn about numpy arrays, a data structure available Python's numpy library that implements a variety of useful functions for analyzing data.

# ## Numpy and Array Basics

# The numpy library is one of the core packages in Python's data science software stack. Many other Python data analysis libraries require numpy as a prerequisite, because they use its array data structure as a building block. The Kaggle Python environment has numpy available by default; if you are running Python locally, the Anaconda Python distribution comes with numpy as well.
# 
# Numpy implements a data structure called the N-dimensional array or ndarray. ndarrays are similar to lists in that they contain a collection of items that can be accessed via indexes. On the other hand, ndarrays are homogeneous, meaning they can only contain objects of the same type and they can be multi-dimensional, making it easy to store 2-dimensional tables or matrices.
# 
# To work with ndarrays, we need to load the numpy library. It is standard practice to load numpy with the alias "np" like so:

# In[2]:


import numpy as np


# The "as np" after the import statement lets us access the numpy library's functions using the shorthand "np."
# 
# Create an ndarray by passing a list to np.array() function:

# In[3]:


my_list = [1, 2, 3, 4]             # Define a list

my_array = np.array(my_list)       # Pass the list to np.array()

type(my_array)                     # Check the object's type


# To create an array with more than one dimension, pass a nested list to np.array():

# In[4]:



second_list = [5, 6, 7, 8]

two_d_array = np.array([my_list, second_list])

print(two_d_array)


# An ndarray is defined by the number of dimensions it has, the size of each dimension and the type of data it holds. Check the number and size of dimensions of an ndarray with the shape attribute:

# In[5]:



two_d_array.shape


# The output above shows that this ndarray is 2-dimensional, since there are two values listed, and the dimensions have length 2 and 4. Check the total size (total number of items) in an array with the size attribute:

# In[6]:


two_d_array.size


# Check the type of the data in an ndarray with the dtype attribute:

# In[7]:


two_d_array.dtype


# Numpy has a variety of special array creation functions. Some handy array creation functions include:

# In[8]:


# np.identity() to create a square 2d array with 1's across the diagonal

np.identity(n = 5)      # Size of the array


# In[9]:


# np.eye() to create a 2d array with 1's across a specified diagonal

np.eye(N = 3,  # Number of rows
       M = 5,  # Number of columns
       k = 1)  # Index of the diagonal (main diagonal (0) is default)


# In[10]:


# np.ones() to create an array filled with ones:

np.ones(shape= [2,4])


# In[11]:


# np.zeros() to create an array filled with zeros:

np.zeros(shape= [4,6])


# ## Array Indexing and Slicing

# Numpy ndarrays offer numbered indexing and slicing syntax that mirrors the syntax for Python lists:

# In[14]:


one_d_array = np.array([1,2,3,4,5,6])

one_d_array[3]        # Get the item at index 3


# In[15]:


one_d_array[3:]       # Get a slice from index 3 to the end


# In[16]:


one_d_array[::-1]     # Slice backwards to reverse the array


# If an ndarray has more than one dimension, separate indexes for each dimension with a comma:

# In[17]:


# Create a new 2d array
two_d_array = np.array([one_d_array, one_d_array + 6, one_d_array + 12])

print(two_d_array) 


# In[18]:


# Get the element at row index 1, column index 4

two_d_array[1, 4]


# In[19]:


# Slice elements starting at row 2, and column 5

two_d_array[1:, 4:]


# In[20]:


# Reverse both dimensions (180 degree rotation)

two_d_array[::-1, ::-1]


# ## Reshaping Arrays

# Numpy has a variety of built in functions to help you manipulate arrays quickly without having to use complicated indexing operations.
# 
# Reshape an array into a new array with the same data but different structure with np.reshape():

# In[21]:


np.reshape(a=two_d_array,        # Array to reshape
           newshape=(6,3))       # Dimensions of the new array


# Unravel a multi-dimensional into 1 dimension with np.ravel():

# In[22]:


np.ravel(a=two_d_array,
         order='C')         # Use C-style unraveling (by rows)


# In[23]:


np.ravel(a=two_d_array,
         order='F')         # Use Fortran-style unraveling (by columns)


# Alternatively, use ndarray.flatten() to flatten a multi-dimensional into 1 dimension and return a copy of the result:

# In[24]:


two_d_array.flatten()


# Get the transpose of an array with ndarray.T:

# In[25]:


two_d_array.T


# Flip an array vertically or horizontally with np.flipud() and np.fliplr() respectively:

# In[26]:


np.flipud(two_d_array)


# In[27]:


np.fliplr(two_d_array)


# Rotate an array 90 degrees counter-clockwise with np.rot90():

# In[28]:


np.rot90(two_d_array,
         k=1)             # Number of 90 degree rotations


# Shift elements in an array along a given dimension with np.roll():

# In[29]:


np.roll(a= two_d_array,
        shift = 2,        # Shift elements 2 positions
        axis = 1)         # In each row


# Leave the axis argument empty to shift on a flattened version of the array (shift across all dimensions):

# In[30]:


np.roll(a= two_d_array,
        shift = 2)


# Join arrays along an axis with np.concatenate():

# In[31]:


array_to_join = np.array([[10,20,30],[40,50,60],[70,80,90]])

np.concatenate( (two_d_array,array_to_join),  # Arrays to join
               axis=1)                        # Axis to join upon


# ## Array Math Operations

# Creating and manipulating arrays is nice, but the true power of numpy arrays is the ability to perform mathematical operations on many values quickly and easily. Unlike built in Python objects, you can use math operators like +, -, / and * to perform basic math operations with ndarrays:

# In[32]:


two_d_array + 100    # Add 100 to each element


# In[33]:


two_d_array - 100    # Subtract 100 from each element


# In[34]:


two_d_array * 2      # Multiply each element by 2


# In[35]:


two_d_array ** 2      # Square each element


# In[36]:


two_d_array % 2       # Take modulus of each element 


# Beyond operating on each element of an array with a single scalar value, you can also use the basic math operators on two arrays with the same shape. When operating on two arrays, the basic math operators function in an element-wise fashion, returning an array with the same shape as the original:

# In[37]:


small_array1 = np.array([[1,2],[3,4]])

small_array1 + small_array1


# In[38]:


small_array1 - small_array1


# In[39]:


small_array1 * small_array1


# In[40]:


small_array1 ** small_array1


# Numpy also offers a variety of [named math functions](https://docs.scipy.org/doc/numpy/reference/routines.math.html) for ndarrays. There are too many to cover in detail here, so we'll just look at a selection of some of the most useful ones for data analysis:

# In[41]:


# Get the mean of all the elements in an array with np.mean()

np.mean(two_d_array)


# In[42]:


# Provide an axis argument to get means across a dimension

np.mean(two_d_array,
        axis = 1)     # Get means of each row


# In[43]:


# Get the standard deviation all the elements in an array with np.std()

np.std(two_d_array)


# In[44]:


# Provide an axis argument to get standard deviations across a dimension

np.std(two_d_array,
        axis = 0)     # Get stdev for each column


# In[45]:



# Sum the elements of an array across an axis with np.sum()

np.sum(two_d_array, 
       axis=1)        # Get the row sums


# In[46]:


np.sum(two_d_array,
       axis=0)        # Get the column sums


# In[47]:


# Take the log of each element in an array with np.log()

np.log(two_d_array)


# In[48]:


# Take the square root of each element with np.sqrt()

np.sqrt(two_d_array)


# Take the dot product of two arrays with np.dot(). This function performs an element-wise multiply and then a sum for 1-dimensional arrays (vectors) and matrix multiplication for 2-dimensional arrays.

# In[49]:


# Take the vector dot product of row 0 and row 1

np.dot(two_d_array[0,0:],  # Slice row 0
       two_d_array[1,0:])  # Slice row 1


# In[50]:


# Do a matrix multiply

np.dot(small_array1, small_array1)


# The package includes a variety of more advanced [linear algebra functions](https://docs.scipy.org/doc/numpy/reference/routines.linalg.html) should you need them.

# ## Wrap Up

# Numpy's ndarray data structure provides many desirable features for working with data, such as element-wise math operations and a variety of functions that work on 2D arrays. Since numpy was built with data analysis in mind, its math operations are optimized for that purpose and are generally faster than what could be achieved if you hand-coded functions to carry out similar operations on lists.
# 
# Numpy's arrays are great for performing calculations on numerical data, but most data sets you encounter in real life aren't homogeneous. Many data sets include a mixture of data types including numbers, text and dates, so they can't be stored in a single numpy array. In the next lesson we'll conclude our study of Python data structures with Pandas DataFrames, a powerful data container that mirrors the structure of data tables you'd find in databases and spreadsheet programs like Microsoft Excel.

# ## Next Lesson: [Python for Data 9: Pandas DataFrames](https://www.kaggle.com/hamelg/python-for-data-9-pandas-dataframes)
# [back to index](https://www.kaggle.com/hamelg/python-for-data-analysis-index)
