#!/usr/bin/env python
# coding: utf-8

# # NumPy - NumericalPython
# <li>NumPy-is a Linear Algebra Library for Python.</li>
# <li>Numpy is executes faster, it has bindings to C libraries.

# In[ ]:


#using numpy 
import numpy as np


# In[ ]:


#Creaing 1-d array

l=[10,20,30]
a=np.array(l)
a


# In[ ]:


#Creating 2-d array

l2=[[10,20,30],[40,50,60],[70,80,90]]

a2=np.array(l2)

a2


# In[ ]:


#Creating array using arange():

a1=np.arange(0,15,2)
a1


# In[ ]:


#Creating array using random

b=np.random.rand(5)   #rand-generates between 0 & 1(Exclusive)
b

#randn,randint also can be used


# In[ ]:


#Getting the dimensions of existing array 

a2.shape


# In[ ]:


#Changing the dimensions of existing array 
a1.reshape(2,4)


# In[ ]:


#Creating array of ones 
o =np.ones((5,3))
o


# In[ ]:


#Creating array of zeros
z=np.zeros(6)
z


# In[ ]:


#Creating identity matrix[Matrix which has equal number of rows and columns also all the non-diagonal elements are zero,also diagonal elments are 1]

i=np.eye(5)
i


# In[ ]:


#linspace-Returns equally spaced elements winthin a range
l=np.linspace(10,50,10)
l


# In[ ]:


#Inbulit arthimetic operations

print(a.min())
print(a.max())
print(a.sum())
print(a.argmax()) #index of maximum element
print(a.argmin()) #index of minimum elment


# ## Indexing and selecting

# In[ ]:


arr=np.random.randint(0,100,15)
arr=arr.reshape(5,3)
arr


# In[ ]:


arr[3][1]  #[][] row and column


# In[ ]:


arr[0,2]  #[row,column]


# In[ ]:


#slicing
arr[2:,0:2]


# In[ ]:


arr


# In[ ]:


#Broadcasting 

arr[:3,]=100
arr


# Comparison

# In[ ]:


[arr<50]


# In[ ]:


arr[arr<50]


# # Operations with array : Performs element-wise operations

# In[ ]:


a=np.random.randint(0,10,20)
a


# In[ ]:


b=np.random.randint(20,30,20)
b


# In[ ]:


a+b


# In[ ]:


a-b


# In[ ]:


a*b


# In[ ]:


a/b


# In[ ]:


np.sqrt(a)


# In[ ]:


np.dot(a,b) #dot product 


# In[ ]:


a


# In[ ]:


np.flip(a) #Reverses array


# Refernce<br>
# 
# <a href="https://numpy.org/doc/1.19/user/absolute_beginners.html">Documentation</a>

# In[ ]:





# In[ ]:




