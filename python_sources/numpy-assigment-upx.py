#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# ### Create a vector with values ranging from 10 to 49

# In[ ]:


vector1 = np.arange(10,49,1)
vector1


# ### Reverse a vector (first element becomes last)

# In[ ]:


vector1_rev = vector1[::-1]
vector1_rev


# ### Create a 3x3 matrix with values ranging from 0 to 8

# In[ ]:


matrix1 = np.arange(0,9,1).reshape(3,3)
matrix1


# ### Find indices of non-zero elements from [1,2,0,0,4,0]

# In[ ]:


a = [1,2,0,0,4,0]
np.nonzero(a)


# ### Create a 3x3x3 array with random values

# In[ ]:


np.random.rand(3,3,3)


# ### Create a 10x10 array with random values and find the minimum and maximum values

# In[ ]:


b = np.random.rand(10,10)
print('min =', np.min(b))
print('max =', np.max(b))


# ### Create a 5x5 matrix with row values ranging from 0 to 4

# In[ ]:


c = np.zeros((5,5))
#c+= np.arange(5)
c+=[0,1,2,3,4]
c


# ### Consider two random array A and B, check if they are equal

# In[ ]:


A = np.random.rand(2,2)
B = np.random.rand(2,2)
#A==B
np.array_equal(A,B)


# ### find the closest value (to a given scalar) in a vector

# In[ ]:


#using vector1
scalar1 = np.random.uniform(0,40)
print(scalar1)
index = (np.abs(vector1 - scalar1)).argmin()
print(vector1[index])


# ### Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value?

# In[ ]:


vector2 =  [1, 2, 3, 4, 5]
new_vector =np.zeros(len(vector2)+(len(vector2)-1)*3)
new_vector[::3+1]=vector2
print(new_vector)


# ### How to find the most frequent value in an array?

# In[ ]:


np.bincount(b).argmax()

