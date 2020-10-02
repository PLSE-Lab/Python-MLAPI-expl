#!/usr/bin/env python
# coding: utf-8

# This is the first part of a series of notebooks about machine learning. I will take the reader through the basic functionality of [NumPy](https://numpy.org). NumPy is the main package for scientific computing in Python. Machine learning is not different from scientific computing in this regard; it requires a lot of computations for which NumPy is a cornerstone.
# 
# As someone who is excited about machine learning, NumPy might not immediately be very exciting, since it might not be obvious at the beginning what the link between this package and machine learning is, especially for someone who is just starting with machine learning. However, as a package which is used a lot in machine learning, it is important to build a good familiarity of NumPy.
# 
# # Why NumPy?
# 
# NumPy provides computational functionalities like arithmetic, mathematical functions, etc. The first question that might come to mind is why do we need this when Python provides many computational functionalities out of the box? There are multiple reasons:
# 
# 1. Python mainly deals with individual numbers, as opposed to NumPy which makes computations on arrays much easier.
# 2. Numbers in python are [objects](https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html#A-Python-Integer-Is-More-Than-Just-an-Integer), making computations, especially on a large number of them, much slower than NumPy, which stores numbers efficiently and employ [CPU special instructions](https://stackoverflow.com/questions/8385602/why-are-numpy-arrays-so-fast) to make computations extremely fast.
# 3. Finally, NumPy provides a plethora of functions that are otherwise not available in Python out of the box. NumPy is actually a computational framework.

# # How Fast is it?
# 
# Calculations in NumPy are usually faster than vanilla Python by orders of magnitude. Check the example below. It generates random a list of numbers (fixing the seed to 0 so the same list is generated every time) then use Python to find the square root of each element in the list. It then uses NumPy to do the same.

# In[ ]:


import random
import math
import numpy as np

random.seed(0)
rand_list = [random.randint(0, 100) for r in range(1000)]

def list_sqrt(list):
    return [math.sqrt(n) for n in list]

get_ipython().run_line_magic('timeit', 'rand_list_sqrt = list_sqrt(rand_list)')


# In[ ]:


np_rand_array = np.array(rand_list)
get_ipython().run_line_magic('timeit', 'np_rand_array_sqrt = np.sqrt(np_rand_array)')


# Notice the 3-fold difference in speed. Furthermore, notice that without NumPy I had to define a function to find the square root of an array, while with NumPy I simply used the `np.sqrt` function.
# 
# The remaining of this notebook will go through the different functionalities of NumPy by example. As a reference, I depended on [NumPy manual](https://numpy.org/doc/1.17/index.html) and [Chapter 2](https://jakevdp.github.io/PythonDataScienceHandbook/#2.-Introduction-to-NumPy) of [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do).

# # Creating NumPy Arrays

# In[ ]:


# From a Python list
array1 = np.array([1, 2, 3, 4, 5])
array1


# In[ ]:


# Print the underlying data type used by the array.
array1.dtype


# In[ ]:


# NumPy's equivalent of `range`
array2 = np.arange(0, 100, 2)
array2


# In[ ]:


# Using a specific data type. See the following page for a list of available types: https://numpy.org/doc/1.17/user/basics.types.html
array3 = np.arange(0, 100, 2, dtype='float32')
array3


# In[ ]:


# Fill an array with zeros
array4 = np.zeros(10, dtype='float64')
array4


# In[ ]:


# Or ones
array5 = np.ones(10, dtype='int32')
array5


# In[ ]:


# Or custom number
array6 = np.full(30, 2.7818)
array6


# In[ ]:


# Or a range between certain numbers, with a certain stride (3 in this case).
array7 = np.arange(0, 100, 3)
array7


# In[ ]:


# Or a range between certain numbers, specifying the total number of numbers instead of stride.
array8 = np.linspace(0, 10, 21) # 21 uniformly-spaced numbers between 0 and 10
array8


# In[ ]:


# Randomly generating an array of floats.
array9 = np.random.random(10)
array9


# In[ ]:


# Or an array of integers between a specific range
array10 = np.random.randint(10, 20, size=50) # 50 random integers between 10 and 20.
array10


# For more information on array creation, see [NumPy's reference](https://numpy.org/doc/1.17/reference/routines.array-creation.html).

# # Multi-Dimensional Arrays
# 
# Dealing with multi-dimensional arrays in NumPy is not much different from dealing with single dimensional arrays. The examples below illustrates how.

# In[ ]:


# Creating a 2-D array of zeros.
array11 = np.zeros((10, 10))
array11


# In[ ]:


# Similarly, the type can be specified.
array12 = np.zeros((10, 10), dtype='int64')
array12


# In[ ]:


# Or fill with ones.
array13 = np.ones((5, 5), dtype='int64')
array13


# In[ ]:


# Creating 3-D arrays is as easy.
array14 = np.ones((3, 3, 3), dtype='int64')
array14


# In[ ]:


# Back to 2-D arrays, you could also convert a Python list of lists into an array.
array15 = np.array([[1, 2, 3],
                    [2, 3, 2],
                    [3, 2, 1]])
array15


# # NumPy's Array Attributes
# 
# Now that we got a taste of how NumPy works, let's get into some details. The main object of NumPy is the the [ndarray object](https://numpy.org/doc/1.17/reference/arrays.ndarray.html). This object has the following [attributes](https://numpy.org/doc/1.17/user/quickstart.html#the-basics) which are useful to understand:
# 
# - **ndim**: The number of dimensions of the array.
# - **shape**: A tuple containing the size of each dimension, e.g. a `2x3` matrix will have a shape of `(2, 3)`.
# - **dtype**: The type of the elements of the array, e.g. `int32`, `float32`, etc.
# - **itemsize**: The size in bytes of each element of the array. For example, for `int32` the size is 4.
# - **size**: The total number of elements in the array. This is equivalent to the multiplication of the elements of `shape`. For example, for a shape of `(2, 3)`, the size is 2 x 3 = 6.
# 
# The example below best illustrates those fields.

# In[ ]:


array16 = np.ones((3, 5), dtype='float64')
print(array16.ndim)
print(array16.shape)
print(array16.dtype)
print(array16.itemsize)
print(array16.size)


# # Indexing
# 
# Indexing is very powerful in NumPy. You can extract single elements, multiple adjacent elements, multiple elements randomly selected, certain columns or rows from matrics, and so on. Again, this is best explained by examples.

# In[ ]:


# I will use this array for the examples of this section.
np.random.seed(0)
array17 = np.random.randint(0, 10, size=(4, 5))
array17


# In[ ]:


# Extract the first row.
array17[0, :]


# In[ ]:


# Or the last row.
array17[-1, :]


# In[ ]:


# Extract the first and third columns.
array17[:, [0, 2]]


# In[ ]:


# Extract the element in the 3rd row and 4th column. This should have been the first example, shouldn't it?
array17[2, 3]


# In[ ]:


# From the first row, extract the third element up until the end.
array17[0, 2:]


# In[ ]:


# Extract the elements in the middle.
array17[1:3, 1:4]


# In[ ]:


# Like indexing for Python lists, you could use negative strides to arrays. The line below extract the first row in reversed order.
array17[0, ::-1]


# For more examples, see [Fancy Indexing](https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html) in Chapter 2 of [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do).

# # Modifying Arrays
# 
# Modifying NumPy arrays is no different than modifying Python lists. However, with the power of indexing as explained in the previous seciton, you can do much more than you can with a Python list.

# In[ ]:


array18 = array17.copy() # Copy the array so we don't modify the original one.
array18


# In[ ]:


array18[0, 0] = 100
array18


# In[ ]:


# Change the first three elements of the first row in one operation.
array18[0, 0:3] = (1000, 2000, 3000)
array18


# In[ ]:


# Change the first and third elements of the fourth column.
array18[[0, 2], 3] = [5000, 6000]
array18


# You get the basic idea. You could try any different combination of indexing and assign values to them in one go. This is really powerful, especially for processing huge arrays as is very common in machine learning.

# # Universal Functions
# 
# The most important feature in NumPy after arrays themselves is universal functions. Don't let the term "universal" confuse you in that they are special functions. They are the same typical computational functions like addition, multiplication, log, sin, cos, etc. They are called universal because they apply to multiple elements at the same time. So, instead of iterating over all the elements of an array, you simply pass the array to those functions and they will do the job. Let's demonstrate this with examples.

# In[ ]:


array19 = np.arange(0, 10)
array19


# In[ ]:


# Find the squares of the numbers.
array20 = np.square(array19)
array20


# In[ ]:


# Let's find the difference between the numbers and their squares.
array21 = np.subtract(array20, array19)
array21


# In[ ]:


# Actually, for substraction and other elementary functions, you could simply
# use the Python operators, but I wanted to illustrate the original function
# which the operator will end up calling.
array21 = array20 - array19
array21


# In[ ]:


# Find the sines for values between 0 and 2*pi.
array22 = np.linspace(0, 2*np.pi, 20)
array22_sin = np.sin(array22)
array22_sin


# In[ ]:


# and the cosines
array22_cos = np.cos(array22)
array22_cos


# In[ ]:


# Having the sines and cosines, we might as well draw the circle!
from matplotlib import pyplot as plt
plt.figure(figsize=(4, 4))
plt.plot(array22_sin, array22_cos)


# There are many more universal functions in NumPy, but this should be enough to introduce the basic idea. For a list of available functions, see [NumPy reference](https://numpy.org/doc/1.17/reference/ufuncs.html#available-ufuncs). Also see [Computation on NumPy Arrays: Universal Functions](https://jakevdp.github.io/PythonDataScienceHandbook/02.03-computation-on-arrays-ufuncs.html) section of Chapter 2 of [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do).

# # Aggregation and Sorting
# 
# NumPy also provides a set of functions for use with aggregation, e.g. min, max, etc. The way you use those functions is not different than universal functions.
# 

# In[ ]:


# Let's use NumPy to sum the numbers from 1 to 100 to see whether Gauss was right: https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss#Anecdotes
array23 = np.arange(1, 101)
np.sum(array23)


# In[ ]:


# To find the average:
np.average(array23)


# In[ ]:


# To find the minimum:
np.min(array23)


# In[ ]:


# To find the maximum:
np.max(array23)


# Let's generate a random set of points and find their centre of mass:

# In[ ]:


array24_x = np.random.randint(-10, 10, size=50) # 50 random integers between 10 and 20.
array24_y = np.random.randint(-10, 10, size=50) # 50 random integers between 10 and 20.
array24_x_centre = np.average(array24_x)
array24_y_centre = np.average(array24_y)

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 6))
plt.plot(array24_x, array24_y, 'o', color='green')
plt.plot(array24_x_centre, array24_y_centre, 'x', color='red')


# Like aggregation, sorting is also easy to do with NumPy.

# In[ ]:


array24 = np.random.randint(0, 100, size=50)
array24


# In[ ]:


# Sort the array.
array25 = np.sort(array24)
array25


# # Comparison on Arrays
# 
# You can apply comparison operations on arrays. For example, assume you have an array containing the daily average temperature and you want to find temperatures above a certain number.

# In[ ]:


array26 = np.random.randint(10, 30, size=50) # random temperatures in Celesius
array26


# In[ ]:


# Comparison operators produce a True/False array.
array26 > 20


# In[ ]:


# By passing in a True/False array to another array, we could extract
# the elements for which we pass True.
array26[array26 > 20]


# In[ ]:


# We could then use a function like count_nonzero on the boolean array
# to find the number of days in whech the temparuter is above 20.
np.count_nonzero(array26 > 20)


# # Broadcasting
# 
# At times, we need to perform operations among arrays of different sizes, or arrays and scalars, and so on. For example, it is very common in statistics to need to [normalize](https://en.wikipedia.org/wiki/Normalization_%28statistics%29) a list of numbers, for which we need to find the mean and then substract it from all numbers, and then divide by the standard deviation. NumPy allows us to do such operations by what is called [broadcasting](https://numpy.org/doc/1.17/reference/ufuncs.html#broadcasting). Let's implement the normalization using NumPy to see how this works.

# In[ ]:


# Let's start by generating an array of 10 random numbers between 0 and 100.
array27 = np.random.random(10)*100
array27


# In[ ]:


# Find the average
array27_mean = np.average(array27)
array27_std = np.std(array27)
print(f"Mean = {array27_mean}")
print(f"Standard Deviation = {array27_std}")


# In[ ]:


# Normalize the array: substract the average from all the elements and divide the
array27_normal = (array27 - array27_mean)/array27_std
array27_normal


# As you can see, by simply substracting a number from an array, NumPy automatically substracted the number from every element of the array; same thing with the division by the standard deviation. This is broadcasting in NumPy. In fact, I cheated a little bit in this section, as when I generated the random list above, I used broadcasting when multiplying the array generated by `np.random.random` by 100; that was also broadcasting.
# 
# Broadcasting is not limited to operation between a 1-D array and a scalar. For example, you can subtract a `[1 x 5]` matrix from a `[5 x 5]` matrix, and NumPy will do the equivalent of duplicating the `[1 x 5]` 5 times row-wise such that it becomes a `[5 x 5]` matrix, then subtract it from the other `[5 x 5]` matrix.
# 
# It might be a little bit complicated at the beginning to understand how broadcasting works, and in fact there are [rules](https://numpy.org/doc/1.17/user/quickstart.html#broadcasting-rules) that I encourage you to read, but the idea is basically that **NumPy will try to make the operands match in size by doing the necessary duplications, and then apply the operation**. For example, in the normalization example above, when we substracted the mean from the array, you could think of it as generating another array containing the mean duplicated across it, and then substracting that array from the original array.
# 
# Broadcasting is extremely useful in machine learning. Usually, you have a huge number of samples that you want to train your model on. To achieve the best performance, the samples are stored in huge matrices and the same operation (whatever it is that need to be computed) is applied to all the rows of the matrix. For example, if we have a million arrays to be normalized, then we could put them all in one huge matrix and then find a vector containing the means and another one containing the standard deviation, then employ broadcasting to efficiently normalize all arrays:
# 

# In[ ]:


# reduce precision to make printing compact
get_ipython().run_line_magic('precision', '2')
array28 = np.random.random((3, 7))*100
array28


# In[ ]:


# Assuming each column represent a sample, find the mean of each column.
array28_mean = np.average(array28, axis=0) # Notice that we have to specify the dimension,
                                           # otherwise the function would average all elements.
array28_mean


# In[ ]:


# Similarly, find the standard deviation.
array28_std = np.std(array28, axis=0)
array28_std


# In[ ]:


# Now let's do only the substraction so we could see what is happening
array28 - array28_mean


# In[ ]:


# Now find the full normalization
(array28 - array28_mean)/array28_std


# As easy as that! Notice that we didn't have to worry about providing the dimensions of the array, all we did was a simple substraction and division and NumPy did the rest.

# # NumPy for MATLAB/Octave Users
# 
# For those of you have already used MATLAB/Octave before, the material presented in this notebook should not be new to you and the important thing is how to map your knowledge of MATLAB/Octave to NumPy. I found [this page](https://numpy.org/doc/1.17/user/numpy-for-matlab-users.html) from NumPy documentation to be useful as a quick cheatsheet. I hope you find it useful too.

# # Conclusion
# 
# I hope the demos in this notebook helped you build a good understanding of NumPy. It is, nevertheless, still a good to do more reading. Consider reading [Chatpre 2: Introduction to NumPy](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html) of the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do). It is also worth navigating NumPy's official [user guide](https://numpy.org/doc/1.17/user/index.html) and [reference](https://numpy.org/doc/1.17/reference/index.html).
