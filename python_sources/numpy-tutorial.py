#!/usr/bin/env python
# coding: utf-8

# ## Numpy Tutorial
# 
# NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays. It is the fundamental package for scientific computing with Python
# 
# ### What is an array
# 
# An array is a data structure that stores values of same data type. In Python, this is the main difference between arrays and lists. While python lists can contain values corresponding to different data types, arrays in python can only contain values corresponding to same data type

# In[ ]:


## initially Lets import numpy

import numpy as np
import pandas as pd


# In[ ]:


my_lst=[1,2,3,4,5]

arr=np.array(my_lst)


# In[ ]:


print(arr)


# In[ ]:


type(arr)


# In[ ]:


## Multinested array
my_lst1=[1,2,3,4,5]
my_lst2=[2,3,4,5,6]
my_lst3=[9,7,6,8,9]

arr=np.array([my_lst1,my_lst2,my_lst3])


# In[ ]:


arr


# In[ ]:


type(arr)


# In[ ]:


## check the shape of the array

arr.shape


# ### Array Indexing 

# In[ ]:


## Accessing the array elements

arr


# In[ ]:


arr[2,2]


# In[ ]:


arr


# In[ ]:


arr[1:,:2]


# In[ ]:


arr[:,3:]


# In[ ]:


arr


# In[ ]:


### Some conditions very useful in Exploratory Data Analysis 

val=2

arr[arr<3]


# In[ ]:


## Create arrays and reshape

np.arange(0,10).reshape(5,2)


# In[ ]:


arr1=np.arange(0,10).reshape(2,5)


# In[ ]:


arr2=np.arange(0,10).reshape(2,5)


# In[ ]:


arr1*arr2


# In[ ]:


np.ones((2,5),dtype=int)


# In[ ]:


## random distribution
np.random.rand(3,3)


# In[ ]:


arr_ex=np.random.randn(4,4)


# In[ ]:


arr_ex


# In[ ]:


import seaborn as sns


# In[ ]:


sns.distplot(pd.DataFrame(arr_ex.reshape(16,1)))


# In[ ]:


np.random.randint(0,100,8).reshape(4,2)


# In[ ]:


np.random.random_sample((1,5))


# In[ ]:




