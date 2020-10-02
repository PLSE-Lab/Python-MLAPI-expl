#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Question 1:** Describe a recursive algorithm for finding the maximum element in a sequence, S, of n
# elements. What is your running time and space usage?

# In[ ]:


#Algorithm MaxElement(S, start)
#Input: S, a sequence, start an integer
#Output: the maximum element in the sequence
if start == length of S then {base case where we have only one element to check}
    return S[start]
else
    max_of_rest = MaxElement(S, start + 1) {recursive call to find out the maximum of the rest of
the sequence after start}
    if S[start] > max_of_rest then
        return S[start]
    else
        return max_of_rest


# Running Time of the algorithm: O(n) as the function is a form of Linear Recursion

# **Question 2:** Draw the recursion trace for the computation of power(2,5), using the traditional function
# implemented below:

# In[ ]:


def power(x, n):
 """Compute the value x**n for integer n."""
 if n == 0:
     return 1
 else:
     return x * power(x, n - 1)


# Recursion Trace: 
# See recTrace1.png

# **Question 3:** Draw the recursion trace for the computation of power(2,18), using the repeated squaring
# algorithm, as implemented below:

# In[ ]:


def power(x, n):
 """Compute the value x**n for integer n."""
 if n == 0:
     return 1
 else:
     partial = power(x, n // 2) # rely on truncated division
     result = partial * partial
     if n % 2 == 1: # if n odd, include extra factor of x
         result *= x
     return result


# See recTrace2.png

# **Question 4:** Give a recursive algorithm to compute the product of two positive integers, m and n, using
# only addition and subtraction.

# In[ ]:


def multiply(m,n):
    if n == 1: 
        return m
    else:
        return m + multiply(m,n-1)
    
#Tests#
print(multiply(5,4)) # 20
print(multiply(7,8)) # 56


# **Question 5:** 5. Modify ch05/experiment_list_append.py to investigate the time taken by append operations for
# DynamicArray (ch05/dynamic_array.py).

# In[ ]:


import sys
from time import time

try:
    maxN = int(sys.argv[1])
except:
    maxN = 10000000
from time import time # import time function from time module

def compute_average(n):
    """Perform n appends to an empty list and return average time elapsed."""
    data = []
    start = time() # record the start time (in seconds)
    for k in range(n):
        data.append(None)
    end = time() # record the end time (in seconds)
    return (end - start) / n # compute average per operation

n = 10
while n <= maxN:
    print('Average of {0:.3f} for n {1}'.format(compute_average(n) * 1000000, n))
    n *= 10


# **Question 6** Create a modified version of DynamicArray (ch05/dynamic_array.py) that takes a parameter,
# resize_factor, which it uses to determine the new size (rather than doubling in the original code -
# self._resize(2 * self._capacity)). Using different values of resize_factor, examine if and
# how the average time to append changes.
# 

# In[ ]:


class DynamicArrayWithResizeFactor(DynamicArray):
    """ A dynamic array class which allows for a custom resize factor. """
    def __init__(self, resize_factor):
        super().__init__()
        self._resize_factor = resize_factor
        
    def append(self, obj):
    """ Modified version of append to use the resize factor. """
        if self._n == self._capacity: # not enough room
        # cast to int to allow for fractional resize_factors, add 1 to make sure
        # capacity always increases
            self._resize(int(self._resize_factor * self._capacity) + 1)
        self._A[self._n] = obj
        self._n += 1


# If we try a small resize factor of 1.1, a sample run looks like
# Average of 100.017 for n 10
# Average of 20.013 for n 100
# Average of 8.505 for n 1000
# Average of 7.105 for n 10000
# Average of 7.683 for n 100000
# Average of 6.593 for n 1000000
# Average of 6.609 for n 10000000
# 
# This is significantly slower than the original resize factor of 2. For a larger value of 4:
# Average of 0.000 for n 10
# Average of 10.006 for n 100
# Average of 2.001 for n 1000
# Average of 1.952 for n 10000
# Average of 2.227 for n 100000
# Average of 2.061 for n 1000000
# Average of 2.020 for n 10000000
# 
# Which appears marginally better than a factor of 2. We can also try 10:
# Average of 49.829 for n 10
# Average of 0.000 for n 100
# Average of 2.004 for n 1000
# Average of 1.602 for n 10000
# Average of 1.646 for n 100000
# Average of 1.662 for n 1000000
# Average of 1.747 for n 10000000
