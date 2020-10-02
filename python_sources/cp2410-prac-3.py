#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# **(R-4.1) Describe a recursive algorithm for finding the maximum element in a sequence, S, of n
# elements. What is your running time and space usage?**

# In[27]:


def MaxNum(s,n):
    for i in range(0,n):
        N=s(i)
        if n.ismax(s):
            print("{} is max".format(N))
        else :
          MaxNum(s,n)
        


# **2. (R-4.2) Draw the recursion trace for the computation of power(2,5), using the traditional function
# implemented below:**

# In[ ]:


def power(x, n):
# """Compute the value x**n for integer n."""
    if n == 0 :
    return 1
    else :
    return x * power(x, n - 1 )


# If power(2,5) with the function being defined as 
# 
#     
# \\(Power(2,5)=>power(2,4)=>power(2,3)=>power(2,2)=>power(2,1)=>power(2,0)\\)
# 
# 
# \\(2*5=10,         2*4=8,             2*3=6,            2*2=4,           2*1=2,            1\\)
# 

# **(R-4.3) Draw the recursion trace for the computation of power(2,18), using the repeated squaring
# algorithm, as implemented below:**

# In[32]:


def power(x, n):
# """Compute the value x**n for integer n."""
    if n == 0:
        return 1
    else:
        partial = power(x, n // 2 ) # rely on truncated division
        result = partial * partial
    if n % 2 == 1: # if n odd, include extra factor of x
        result *= x
        return result


# \\(Power(2,18)=> Power(2,9)=> Power(2,4)=> Power(2,2)=> Power(2,1)=> Power(2,1)\\)
# \\(512*512=262144, 16*16*2=8,4*4=16,2*2=4,1*1*2=2,1\\)
# 

# **(C-4.12) Give a recursive algorithm to compute the product of two positive integers, m and n, using
# only addition and subtraction.**

# In[43]:


def product(m,n):
    if n ==1:
        return m
    else:
        m += product(m,n) 
        return m


# **Q5. Modify ch05/experiment_list_append.py to investigate the time taken by append operations for
# DynamicArray (ch05/dynamic_array.py).**

# In[51]:


import sys
from time import time
from dynamic_array import DynamicArray
try :
    maxN = int(sys.argv[ 1 ])
except :
    maxN = 10000000
from time import time # import time function from time module
def compute_average(n):
#     """Perform n appends to an empty list and return average time elapsed."""
    data = DynamicArray()
    start = time() # record the start time (in seconds)
    for k in range(n):
        data.append( None )
        end = time() # record the end time (in seconds)
        return (end - start) / n # compute average per operation
        n = 10
        while n <= maxN:
            print( 'Average of {0:.3f} for n {1}' .format(compute_average(n) * 1000000 , n))
            n *= 10


# **Q6. Create a modified version of DynamicArray (ch05/dyanmic_array.py) that takes a parameter,
# resize_factor, which it uses to determine the new size (rather than doubling in the original code -
# self._resize(2 * self._capacity) ). Using different values of resize_factor, examine if and
# how the average time to append changes.**
