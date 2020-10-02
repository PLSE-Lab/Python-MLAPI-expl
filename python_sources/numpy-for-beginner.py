#!/usr/bin/env python
# coding: utf-8

# In this kernel I will be covering the basics of Numpy.This is a kernel in process and i will keep on updating this in the coming days.If you like my work please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Creating Arrays**

# In[ ]:


my_list1=[1,2,3,4]
my_list1


# In[ ]:


# Making an array using a list
my_array1=np.array(my_list1)
my_array1


# In[ ]:


my_list2=[11,22,33,44]


# In[ ]:


#Combining the two lists 
my_lists=[my_list1,my_list2]


# In[ ]:


#Creating a 2D array
my_array2=np.array(my_lists)
my_array2


# In[ ]:


my_array2.shape


# In[ ]:


#Finding out the datatype of the array 
my_array2.dtype


# In[ ]:


#Creating an array of all zeros
my_zeros_array=np.zeros(5)
my_zeros_array                   


# In[ ]:


#Checking the datatype of Zero Array 
my_zeros_array.dtype


# In[ ]:


#Creating an array of ones 
np.ones([5,5])


# In[ ]:


#Creating empty arrays 
np.empty(5)


# In[ ]:


#Creating an identity matrix
np.eye(5)


# In[ ]:


np.arange(5)


# In[ ]:


#Creating an array starting with 5 and enfing with 50 with a space of 2
np.arange(5,50,2)


# **Using arrays and scalars **

# In[ ]:


arr1=np.array([[1,2,3,4],[8,9,10,11]])
arr1


# In[ ]:


#Multiplying two arrays 
arr1*arr1


# In[ ]:


#Subtraction on arrays
arr1-arr1


# In[ ]:


#Scalar multiplication 
1/arr1


# In[ ]:


#Exponenetial operation
arr1**3


# **Indexing Arrays**

# 
