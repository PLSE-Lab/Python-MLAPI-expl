#!/usr/bin/env python
# coding: utf-8

# # Say Hello to Numpy:
# 
# 
# Let's start our journey with numpy. NUMPY is a python library , capable of doing array processing with an ease. 
# 
# Numpy is also capable of performing any kind of array operation with a lightining speed. 
# 
# This is much needed tool to acomplish day to day task which requires multi-dimentional array. 
# 
# Numpy is also useful for linear algebra. 
# 
# As a data scientist we daily deal with arrays :)

# In[ ]:


import numpy as np # linear algebra


# # 1D Array or Vector

# In[ ]:


arr = np.array([1,2,3])
print(type(arr))
print(arr)


# Above example is creation of numpy array with the list.
# 
# Class of numpy array is ndarray

# In[ ]:


print("Elemment type: ",arr.dtype)


# dtype describes the type of the elements in the array. For interger elements, int64 the default type 

# In[ ]:


print("Number of dimensions:", arr.ndim)


# ndim returns the number of dimensions of the array

# In[ ]:


print("Shape of an array:",arr.shape)


# arr.shape returns a tuple (# of rows, # of cols). In above example it returns a tuple of (3,) it means it has 3 rows and 1 column

# In[ ]:


print("Size of an array:",arr.size)


# arr.size returns the total number of elements in the array. In above example we have only 3 elements.

# # 2D Array or Matrix

# In[ ]:


matrix = np.array([[1,2,3], [4,5,6]])
print(matrix)


# In[ ]:


print("Element type:",matrix.dtype)
print("Number of dimension:",matrix.ndim)
print("Shape of matrix:",matrix.shape)
print("Size of matrix:",matrix.size)


# # Float / Char / String Matrix
# 
# > We can also create matrix or vector of differnet datatype like float, char and string
# 

# In[ ]:


f_matrix = np.array([[1.0,2.0], [3.0, 4.0]])
print(f_matrix.dtype)


# In[ ]:


c_matrix = np.array([['a','b'], ['c', 'd']])
print(c_matrix.dtype.name)


# In[ ]:


s_matrix = np.array([["hello", "Hi"], ["Google", "Facebook"]])
print(s_matrix.dtype.name)


# # Mixed datatype
# 
# > Can we create an numpy array which contains mixed datatype ?

# In[ ]:


mix = [1, 2.0]
print(mix)


# Above list contains two elements one is integer and another is float. In python a list can have any object. Let's try to create a numpy array with the above list.

# In[ ]:


print(np.array(mix))


# All elements of a numpy array has same datatype. 
# 
# So when numpy founds one element integer and another float. It converted integer into float and created an array

# In[ ]:


mix = [1, 2.0, '3']
print(mix)
print(np.array(mix))


# In this example, we have three different datatype mixed together. 
# 
# So numpy converted integer and float into string and created an array.

# # Smart ways to create ndarray
# 
# > * np.ones : **np.ones** function creates an array of given shape and fills all the element with 1. 
# > * np.zeros : **np.zeros** function creates an array of given shape and fills all the element with 0. 
# > * np.random.random : **np.random.random** function creates an array of given shape and fills all the element with random values.
# > * np.arange: **np.arange** generates a sequence of numbers and returns a single dimensional array. It takes stepsize to generate next number.
# > * np.linespace: **np.linespace** generates evenly spaced numbers over a specified interval.

# In[ ]:


only_ones = np.ones((2,3))
print(only_ones)
print(only_ones.dtype.name)


# In[ ]:


only_ones_of_integer = np.ones((2,3), dtype=np.int16)
print(only_ones_of_integer)
print(only_ones_of_integer.dtype.name)


# **specify datatype :**  we can specify the datatype of numpy array by using dtype. 
# 
# By default it take float64

# In[ ]:


only_zeros = np.zeros((2,3))
print(only_zeros)


# In[ ]:


random_arr = np.random.random((2,3))
print(random_arr)


# In[ ]:


seq = np.arange(5)
print(seq)


# **np.arange** generates the values including start and excluding stop by using step to generate next number.
# 
# By default start is 0, step is 1.
# 
# So for np.arange(5) , it generated numbers by including 0 and excluding 5 by adding 1 as a step.

# In[ ]:


np.arange(10,50,5)


# In the above example, start is 10 and end is 50 and step is 5.
# 
# arange generated an array starting from 10 , excluding 50 by incrementing 5. 

#   

# **Can we generate numbers from 5 to 1 by using np.arange ?**

# In[ ]:


np.arange(5,0,-1)


# In the above example, we are trying to generate numbers in reverse order from 5 to 1. 
# 
# To achive this, we have given start as 5, end as 0 and step is -1. So after generating 5 , it will add the step to get the next number. But here we are using the -1, so 5 -1 = 4. 
# 
# One more thing to notice here is we use 0 as an end, because np.arange excludes the end. 

# In[ ]:


seq = np.linspace(0,10)

print("Number of elements:",seq.size)
print()
print(seq)


# **np.linspace** generates evenly spaced numbers over specified Start and End as input. 
#  
# Number of elements to be generated is 50 by default.

# In[ ]:


start = 5
end = 10
num_of_elements = 10
np.linspace(start, end, num_of_elements)


# In above example, we are trying to generate 10 evenly spaced numbers from 5 to 10 .

# # Play with the shape of Array
# 
# > In this section we will see the tricks of array re-shaping. 
# 
# For example convert 1D array to 2D array or convert 2D array to 1D

# In[ ]:


vector = np.random.random((10,1))
print(vector.shape)
print(vector)


# vector is a random 1D array, that has a shape of 10 rows and 1 column

# In[ ]:


matrix = vector.reshape(2,5)
print(matrix.shape)


# Re-shaped vector into matrix by using reshape() function and specifying the new shape of 2 rows and 5 columns

# In[ ]:


print(matrix)


# In[ ]:


matrix = vector.reshape(2,-1)
print(matrix.shape)


# reshape() function also works with -1, we can pass -1 if we are not sure about that dimension. 
# 
# For exammple : if we are not sure about how many columns, than we can pass -1.  
#         
#                So reshape(2,-1) convert vector into matrix that has 2 rows, reshape() equally divides the number of elemnts into 2 chunks and figures out the number of columns.

# In[ ]:


print(matrix)


# In[ ]:


print(matrix.reshape(10,1))


# With same trick we can convert matrix(2D array) to vector(1D array).
# 
# As we know, matrix has 2 rows and 5 columns, so 2 * 5 = 10 , passing 10 rows and 1 columns as an input to reshape() function.

# In[ ]:


print(matrix.reshape(-1,1))


# We can also use the -1 trick to convert multi-dimensional to 1D array. 
# 
# We know that 1D array has only 1 column, we can fix that and pass rows as -1.

# In[ ]:


flatten_array = matrix.reshape(-1)
print(flatten_array.shape)


# Another way to flatten the multi-dimensional array is reshape(-1).
# 

# In[ ]:


print(flatten_array)


# # Array indexing tricks
# 
# > Array index starts from zero 

# In[ ]:


arr = np.arange(1,11)
arr


# In[ ]:


print("Access the first element of an array :",arr[0])
print("Access the fifth element of an arary :",arr[4])
print()
print("Access the last element of an array: ",arr[-1])
print("Access the second element of an array: ",arr[-2])


# First element start with index 0, second with 1 and so on. 
# 
# -ve index starts from the last element by starting -1, -2 and so on.

# In[ ]:


start_index = 3
end_index = 7

print(arr[ start_index : end_index])


# **arr[3:7]** is a way of selecting sub part of an array. 
# 
# We need to specify the start and end (excluding) index. By default index steps by 1. 
# 
# 
# 

# In[ ]:


start_index = 0
end_index = 10
step_by = 2
print(arr[ start_index : end_index: step_by])


# In above example we are finding the subset of array by passing starting index 0 and end index 10 and by stepping 2 index.
# 
# So index will be 0, 2, 4, 6, 8

# In[ ]:


matrix = np.arange(1,17).reshape(4,4)
print(matrix)


# In[ ]:


print("Element at first row and first column of matrix:",matrix[0,0])
print("Element at last row and last column of matrix:",matrix[3,3])


# In[ ]:


print("Second row of the matrix:")
matrix[1]


# In[ ]:


print("First to third rows of the matrix:")
matrix[0:3]


# In[ ]:


print("Second column of the matrix:")
matrix[:,1]


# In[ ]:


print("First to Third columns of the matrix:")
matrix[:,0:3]


# In[ ]:


print("Find elements of second and third column and second and third row of the matrix")
matrix[1:3, 1:3]


# ## Boolean indexing or masking 

# In[ ]:


matrix


# In[ ]:


mask = matrix > 10
mask


# In[ ]:


matrix[mask]


# In[ ]:





# In[ ]:


even_element_mask = matrix % 2 == 0
even_element_mask


# In[ ]:


matrix[even_element_mask]


# In[ ]:





# # Matrix operations
# 
# > * transpose
# > * element wise operations
# > * dot product
# > * inverse
# > * stacking ( horizontally or vertically )
# 
# 

# In[ ]:


matrix = np.arange(1,11).reshape(2,-1)
matrix


# In[ ]:


transposed_matrix = matrix.T
transposed_matrix


# **Matrix transpose** is changing rows to columns and columns to rows.  
# 
# As we can see that rows of matrix becomes columns and columns becomes rows after applying transpose. 

# In[ ]:


matrix + 10


# In[ ]:


matrix - 10


# In[ ]:


matrix * matrix


# In[ ]:


print("Matrix:",matrix.shape)
print("Transposed matrix: ",transposed_matrix.shape)

matrix.dot(transposed_matrix)


# In[ ]:





# In[ ]:


dummy= np.array([[1,2],[3,4]])
dummy


# In[ ]:


np.linalg.inv(dummy)


# In[ ]:





# # Stacking and Splitting:

# ## 1. Horizontal Stacking:
# > Stacks array horizontally, column-wise

# In[ ]:


h_stacked = np.hstack([matrix, matrix])
h_stacked


# ## 2. Vertical Stacking:
# > Stacks array vertically, row-wise

# In[ ]:


v_stacked = np.vstack([matrix, matrix])
v_stacked


# ## 3. Horizontal splitting:
# > Splits array into multiple sub-array columns-wise (horizontally). We also need to specify number of splits.

# In[ ]:


arr1, arr2 = np.hsplit(h_stacked, 2)
print("Array1:")
print(arr1)

print("\nArray2:")
print(arr2)


# ## 4. Vertical splitting:
# > Splits array into multiple sub-array row-wise (vertically). We also need to specify number of splits.

# In[ ]:


arr1, arr2 = np.vsplit(v_stacked, 2)
print("Array1:")
print(arr1)

print("\nArray2:")
print(arr2)


# In[ ]:





# # Query?
# > * np.all : **np.all** test if all of the elements of any array along with given axis evaluates to True returns True, else False. It like logical AND operation. 
# 
# > * np.any : **np.all** test if any of the array elements of any array along with given axis evaluates to True returns True, else False. It like logical OR operation.
# 
# > * np.nonzero : **np.nonzero** retuns the number of indices of the elements that are non-zero. Returns two arrays, one for row indices another for column indices. We can also specify the condition.
# 
# > * np.where : **np.where** returns elements choosen from X or Y after evaluating the given condition. If X or Y is not given results will be same as np.nonzero().

# In[ ]:





# ## np.all

# In[ ]:


arr = np.array([[1,1,0,0],[1,1,0,0],[1,1,0,0],[1,1,0,0] ])
print(arr)


# In[ ]:


np.all(arr==0)


# In[ ]:


np.all(arr==0, axis=0)


# In[ ]:


np.all(arr==0, axis=1)


# In[ ]:





# ## np.any

# In[ ]:


np.any(arr==0)


# In[ ]:


np.any(arr==0, axis=0)


# In[ ]:


np.any(arr==0, axis=1)


# In[ ]:





# ## np.nonzero

# In[ ]:


row_index, col_index = np.nonzero(arr)
print(row_index, col_index)


# In above example, we have passed the array arr to np.nonzero(). It will check if an element is non-zero or not. If yes, returns the row and column indices of all the elements who satisfies.
# 
# Try to print the elements of returned indices to verify. 

# In[ ]:


for r, c in zip(row_index, col_index):
    print("arr[{0}][{1}] = {2}".format(r,c, arr[r][c]))


# np.nonzero with a specific condition

# In[ ]:


row_index, col_index = np.nonzero(arr==0)
print(row_index, col_index)


# In[ ]:


for r, c in zip(row_index, col_index):
    print("arr[{0}][{1}] = {2}".format(r,c, arr[r][c]))


# ## np.where:

# In[ ]:


np.where(arr==0)


# In the above example we want to find the indices of all the elements which has a value zero. Above results are same as np.nonzero()

# In[ ]:


np.where(arr==0, "zero", "non_zero")


# Now are using "zero" and "non_zero" conditional values along with the condition. As a result we got a new array filled with specified values. 
# Values X ("zero") will be chosen for those elements which satisfies the condition , else value Y ("non-zero") will. 
# 
# 
# 
# Not only scaler values, we can pass arrays also in place of X or Y. In that case element will be chosen from the same index.

# In[ ]:


np.where(arr==0, arr, -1)


# In[ ]:





# # Ordering :
# > * max: **np.max** returns the element of the array that has the maximum value. We can also specify the axis.
# 
# > * min: **np.min** returns the element of the array that has the minimum value. We can also specify the axis. 
# 
# > * np.sort: **np.sort** returns the sorted array in ascending order. We can perform the sorting along with axis also. 
# 
# > * np.argmax: **np.argmax** returns the index of the maximum element. 
# 
# > * np.argmin: **np.argmin** returns the index of the minimum elememt. 
# 
# > * np.argsort: **np.argsort** returns the sorted array of indices.  

# In[ ]:


data =[[11, 0, 3, 4], [34, 5, 1, 9], [-9, 5, 3, 6]]
X = np.array(data)
X


# In[ ]:





# ## max:

# In[ ]:


X.max() # maximum element of the array


# In[ ]:


X.max(axis=0)  # Column-wise maximum elements 


# In[ ]:


X.max(axis=1) # Row-wise maximum elements


# In[ ]:





# ## min:

# In[ ]:


X.min() # miniumn element of the array


# In[ ]:


X.min(axis=0) # Column-wise minimum elements of the array


# In[ ]:


X.min(axis=1) # Row-wise minimum elements of the array


# In[ ]:





# ## np.sort:

# In[ ]:


X


# In[ ]:


np.sort(X) # sort array elements in ascending order. By default it's sort the array row-wise


# In[ ]:


-np.sort(-X) # sort array element in descending order.


# In[ ]:


np.sort(X, axis=0) # sort elements column-wise


# In[ ]:


np.sort(X, axis=1) #Sort elements row-wise


# In[ ]:





# ## np.argmax:

# In[ ]:


X


# In[ ]:


np.argmax(X) # index of the maximum element of the array


# In[ ]:


np.argmax(X, axis=0) # indices of the maximum elements of the array column-wise


# In[ ]:


np.argmax(X, axis=1) # indices of the maximum elements of the array row-wise


# In[ ]:





# # np.argmin:

# In[ ]:


X


# In[ ]:


X.reshape(-1)


# In[ ]:


np.argmin(X) # returns the index of the minimum element of the array. If axis is not given , it flattens the array first and finds the index. 


# In[ ]:


np.argmin(X, axis=0) # returns the indices of the minimum elements of the array column-wise


# In[ ]:


np.argmin(X, axis=1) # returns the indices of the minimum elements of the array row-wise


# In[ ]:





# ## np.argsort:

# In[ ]:


X


# In[ ]:


np.argsort(X) # returns the array of indices after sorting the elements in ascending order. 


# In[ ]:


np.argsort(X, axis=0) # returns the array of indices after sorting the elements in ascending order column-wise


# In[ ]:


np.argsort(X, axis=1) # returns the array of indices after sorting the elements in ascending order row-wise


# In[ ]:





# # Statistics:
# > * mean: **np.mean** computes the mean along with the specified axis. By default for the flattened array.
# 
# > * median: **np.median** computes the median along with the specified axis. By default for the flattened array. 
# 
# > * variance: **np.var** computes the variance along with the specified axis. By default if axis is not given, variance is calculated for the flattened array.
# 
# > * standard deviation: **np.std** computes the stardard deviation along with the specified axis. By default if axis is not given, standard deviation is calculated for the flattened array.
# 

# In[ ]:


X = np.arange(1, 11).reshape(5,2)
X


# ## mean:
# \begin{equation*}
# \bar{x} = \left( \sum_{i=0}^n x_i \right) / n
# \end{equation*}

# In[ ]:


np.mean(X) # returns mean of the elements on flattened array


# In[ ]:


np.mean(X, axis=0) # returns the mean of the elements column-wise


# In[ ]:


np.mean(X, axis=1) # returns the mean of the elements row-wise


# ## median: 
# To find median, sort the array in ascending order first. Then return the middle element if array has odd number of elements. In case of even number of elements, return the average of two middle elements.

# In[ ]:


np.median(X) # returns median of the elements on flattened array


# In[ ]:


np.median(X, axis=0) # returns the median of the elements column-wise


# In[ ]:


np.median(X, axis=1) # returns the median of the elements row-wise


# ## variance:
# \begin{equation*}
# \sigma ^ 2 =  \sum_{i=0}^n \left(x_i - \bar{x} \right)^2 / n
# \end{equation*}

# In[ ]:


np.var(X) # returns variance of the elements on flattened array


# In[ ]:


np.var(X, axis=0) # returns the variance of the elements column-wise


# In[ ]:


np.var(X, axis=1) # returns the variance of the elements row-wise


# ## standard deviation:
# \begin{equation*}
# \sigma  =  \sqrt{ \sum_{i=0}^n \left(x_i - \bar{x} \right)^2 / n }
# \end{equation*}

# In[ ]:


np.std(X) # returns standard-deviation of the elements on flattened array


# In[ ]:


np.std(X, axis=0) # returns standard-deviation of the elements column-wise


# In[ ]:


np.std(X, axis=1) # returns standard-deviation of the elements row-wise


# In[ ]:




