#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra


# Any results you write to the current directory are saved as output.


# # 1. What is Numpy

# * NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays
# * NumPy performs computations faster than normal methods

# # 2. The NumPy ndarray

# # <span style="font-size:18px">2.1 Creating ndarrays</span>

# * **list to array**** - below we see how list is converted to numpy array

# In[ ]:


data = [1,2,3,4,5]
arr = np.array(data)
arr


# * **Nested list to array** : we can even create nested list to array, as you see that we get an array where the lists inside the lists are preserved

# In[ ]:


data = [[1,2,3,4],[5,6,3,2]]
arr = np.array(data)
print(arr.shape)
arr


# * **ndim** : but what is the dimension of the array that we have here, lets see it with ndim. As we can see it's a 1d numpy array

# In[ ]:


arr.ndim


# * ***shape*** : So we know the content of the array and the dimension of the array , let's see what is the shape. From shape we can see that array has 2 rows and 4 columns

# In[ ]:


arr.shape


# * **dtype** : Numpy holds data of various data types , it could be float64 or int64 or any other. dtype function helps us find the data type

# In[ ]:


arr.dtype


# * ***np.zeros*** : sometimes we need a numpy array of zeros , for this we can use numpy.zeros function . in argument we supply the shape, datatype, or the order - ** numpy.zeros(shape,dtype=float,order='C')**
# > * shape argument can be integer or tuple of ints
# > * the default datatype argument is float64
# > * order argument tells Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.

# In[ ]:


arr = np.zeros(10)
print(arr.shape) 
print(arr.dtype)
print(arr.ndim)
arr


# In[ ]:


arr = np.zeros((2,3))
print(arr.shape)
print(arr.dtype)
print(arr.ndim)
arr


# In[ ]:


arr = np.zeros((2,3,2))
print(arr.shape)
print(arr.dtype)
print(arr.ndim)
arr


# * ***np.ones*** : one could call it np.zeros twin , it pretty much does the same thing. although it does but 1 in place of zeros. so we get array of 1's insted of 0's ***numpy.ones(shape, dtype=None, order='C'): *** 
# > * parameters are same as ones used for np.zeros

# In[ ]:


arr = np.ones((2,3,2))
print(arr.shape)
print(arr.dtype)
print(arr.ndim)
arr


# * ***np.arange*** : gives us an array of numbers starting from 0 and going till one less than the value given as parameter.so np.arrange(5) would give array([0,1,2,3,4])
# > * all details here - https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html

# In[ ]:


np.arange(5)


# * Hungry for some more cool built in function - I have a screenshot list

# ![1.png](attachment:1.png)

# #  <span style="font-size:18px">2.2Data Types in NumPy</span>

# * we can decide the datatype of a numpy array using dtype , as shown below

# In[ ]:


arr= np.array([1,2,3],dtype=np.float64)
arr


# * Now numpy is cool and gels with many datatypes, here's a screenshot of all its friends

# ![2.png](attachment:2.png)

# ![3.png](attachment:3.png)

# * Numpy wont mind if you change its dtype, infact will help you out with astype()

# In[ ]:


arr = np.array([1,2,3])
print(arr.dtype)
new_arr = arr.astype(np.float64)
print(new_arr.dtype)


# # <span style="font-size:18px">Maths with NumPy</span>

# * NumPy loves maths, that's the best part about it, let's see few examples

# > * lets multiply numpy - 

# > * notice how arr*arr results in each element getting squared

# In[ ]:


arr = np.array([[1,2],[3,4]])
print(arr*arr)
print('________________________')
print(arr)


# > * below we multiply two different numpy array , notice how 1 is multiplied by 2 to get 2 and 2 is multiplied by 4 to result in 8

# In[ ]:


arr1 = np.array([1,2])
arr2 = np.array([2,4])
print(arr1*arr2)


# > * Below we mulitply numpy array by a scalar, notice how each element is multiplied by scalar value, 2 in this case

# In[ ]:


arr = np.array([1,2])
print(arr*2)


# > * Below we multiply two  numpy arrays of different shapes

# In[ ]:


arr1 = np.array([1,2])
arr2 = np.array([2])
print(arr1*arr2)
print(arr1.shape)
print(arr2.shape)


# > * over here in below example numpy got pissed and refused to give result. after some digging around finally numpy agreed to share why it was pissed. here's the reply - <br><br>
# ***When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions, and works its way forward. Two dimensions are compatible when
# they are equal, or
# one of them is 1***

# In[ ]:


arr1 = np.array([1,2])
arr2 = np.array([2,3,5])

print(arr1.shape)
print(arr2.shape)
print(arr1*arr2)


# > ok now lets subtract numpy

# > * Notice how each element is subtracted , 2-1=1 ; 3-2 = 1; 4-3=1

# In[ ]:


arr = np.array([1,2,3])
arr1 = np.array([2,3,4])
print(arr1-arr)


# > * notice how we can't subtract numpy of different shape, unless of course if one of the shapes is 1

# In[ ]:


arr = np.array([1,2])
arr1 = np.array([2,3,4])
print(arr1-arr)


# > * notice how 1 is subtracted from each element

# In[ ]:


arr = np.array([1])
arr1 = np.array([2,3,4])
print(arr1-arr)


# * Lets see how good is numpy in deviding

# > * notice how each element is devided by a scalar

# In[ ]:


arr = np.array([2,4,6])
newArr = arr/2
print(newArr)


# * numpy is very compititive and loves comparison, lets see this below with an example

# > * notice how below when ar1 is compared to arr2 as arr1>arr2 we get list of [False True True] , this is because each element in both numpy array are being compared. 1 > 1 results in False, 2>1 results in True, 3>1 results in true. arr1[0] is compared with arr2[0] , arr1[1] is compared with arr2[1], arr1[2] is compared with arr2[2]

# In[ ]:


arr1 = np.array([1,2,3])
arr2 = np.array([1,1,1])
print(arr1>arr2)


# #  <span style="font-size:18px">2.3 Slicing in numpy</style>

# > * when we slice [x,x+n] we get results including x and excluding x+n
# > * when we slice [x:] we get all values including x 
# > * when we slice [:x] we get all values till x and excluding x
# > * when we slice [:-x] we get all values from strting other than the last x values.
# > * when we slice [-x:] we get last x values
# > * when we slice [x:-y] we get all values starting from x with x included till last y values

# In[ ]:


arr = np.arange(10)
arr[5:10]


# In[ ]:


arr[2:]


# In[ ]:


arr[:2]


# In[ ]:


arr[:-2]


# In[ ]:


arr[-2:]


# In[ ]:


arr[2:-3]


# In[ ]:


arr[-2:5]


# #  <span style="font-size:18px">2.4 Numpy axes</style>

# > * we will look at 2 dimensional numpy since this is most common
# ![4.png](attachment:4.png)

# In[ ]:


arr = np.array([[1,2,3],[2,3,4],[5,4,3],[1,1,1]])
print(arr.shape)
print(arr.ndim)
arr


# In[ ]:


arr[0]


# > * boolean indexing in numpy

# In[ ]:


arr = np.array([1,2,3,4,5,6])
arr[arr > 2]


# In[ ]:


arr[arr > 2 | arr < 4]


# #  <span style="font-size:18px">2.5 Transposing Arrays</style>

# In[ ]:


arr = np.arange(15).reshape((3,5))
arr


# In[ ]:


arr.T


# In[ ]:





# #  <span style="font-size:18px">2.6 Universal Functions</style>

# In[ ]:


arr = np.arange(10)
np.sqrt(arr)


# In[ ]:


np.exp(arr)


# In[ ]:


np.random.rand(4)


# In[ ]:


arr1 = np.array([1,2,3,4])
arr2 = np.array([1,3,2,4])
np.maximum(arr1,arr2)


# In[ ]:


np.where(arr1>3,arr1,arr2)


# > * Here are some other methods we can use-
# ![5.png](attachment:5.png)
# ![6.png](attachment:6.png)
# ![7.png](attachment:7.png)

# #  <span style="font-size:18px">2.7 Mathemetical and Statistical Methods</style>

# In[ ]:


arr = np.arange(5)
print(arr.sum())
print(arr.mean())


# In[ ]:


arr = np.array([[1,2,3],[1,2,5]])
print(arr.sum(axis=1))
print(arr.sum(axis=0))
print(arr.mean(axis=1))
print(arr.mean(axis=0))
arr


# In[ ]:


arr = np.array([2,3,4,1,5])
arr.sort()
arr


# In[ ]:


arr = np.array([3,2,4,5,2,3,4])
np.unique(arr)


# In[ ]:





# In[ ]:





# In[ ]:




