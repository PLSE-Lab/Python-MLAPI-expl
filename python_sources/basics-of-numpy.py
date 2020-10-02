#!/usr/bin/env python
# coding: utf-8

# **Numpy = Numerical Python**
# 1. Numpy : http://www.numpy.org/
# 1. Installation with Python : !pip install numpy
# 1. Installation with Anaconda : !conda install numpy
# 
# Source : https://docs.scipy.org/doc/numpy-1.15.4/user/basics.html

# * a=np.array([]) --> syntax for creating an array
# * a.dtype --> data type of the lements
# * a.size --> tells you number of elements in the array.
# * a.ndim --> Gives the dimension of the array
# * a.shape --> Gives the elements in the each demension

# In[ ]:


# importing numpy
import numpy as np


# **Datatypes in numpy**
# 1. bool_ = Boolean (True or False) stored as a byte
# 1. int_ = Default integer type (same as C long; normally either int64 or int32)
# 1. intc = Identical to C int (normally int32 or int64)
# 1. intp = Integer used for indexing (same as C ssize_t; normally either int32 or int64)
# 1. int8 = Byte (-128 to 127)
# 1. int16 = Integer (-32768 to 32767)
# 1. int32 = Integer (-2147483648 to 2147483647)
# 1. int64 = Integer (-9223372036854775808 to 9223372036854775807)
# 1. uint8 = Unsigned integer (0 to 255)
# 1. uint16 = Unsigned integer (0 to 65535)
# 1. uint32 = Unsigned integer (0 to 4294967295)
# 1. uint64 = Unsigned integer (0 to 18446744073709551615)
# 1. float_ = Shorthand for float64.
# 1. float16 = Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
# 1. float32 = Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
# 1. float64 = Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
# 1. complex_ = Shorthand for complex128.
# 1. complex64 = Complex number, represented by two 32-bit floats (real and imaginary components)
# 1. complex128 = Complex number, represented by two 64-bit floats (real and imaginary components)

# In[ ]:


np.int(2)


# In[ ]:


np.int(2.1)


# In[ ]:


np.int(2.8)


# In[ ]:


# Explore all the different datatypes


# In[ ]:





# In[ ]:





# In[ ]:


a=np.array([0,1,2,3,4,5])
a


# In[ ]:


print('The array is ',a,'and the type is',type(a),'and the elements datatype is',a.dtype)
a = np.array([0,1,2,3.0,4,5])
print('The array is ',a,'and the type is',type(a),'and the elements datatype is',a.dtype)
a = np.array([True,False])
print('The array is ',a,'and the type is',type(a),'and the elements datatype is',a.dtype)


# In[ ]:


# Setting datatype for the elements of an array.
np.array([0, 1, 2], dtype=np.uint8)


# In[ ]:


# Type casting to arrays
a=np.array([0,1,2,3,4,5])
a.astype(float) 


# In[ ]:


print('Demension of an array is',a.ndim)


# In[ ]:


b = np.array([[1,2,3],
              [4,5,6]])
print('The array is ',b,'and the type is',type(b),'and the elements datatype is',b.dtype,'with demensions',b.ndim)


# In[ ]:


c = np.array([[[1,2,3],[4,5,6]],
             [[7,8,9],[10,11,12]]])
print('The array is ',c,'and the type is',type(c),'and the elements datatype is',c.dtype,'with demensions',c.ndim)


# In[ ]:


print('Size of the array is',c.size)


# In[ ]:


d=np.array([[3,4,5,6],
           [7,8,9]])
# If the values are in consistent then it takes each block as one list of type object.
d.dtype


# In[ ]:


e=np.array(['a','b','c'])
e.dtype


# In[ ]:


# arange can only generate a single dimensional array.
f=np.arange(5)
print(f)
g=np.arange(5,55)
print(g)
h=np.arange(5,2)
print(h)
i=np.arange(5,105,5)
print(i)


# In[ ]:


j=np.arange(0,100,5).reshape(5,4)
print(j)


# In[ ]:


k=np.arange(0,100,5).reshape(2,2,5)
print(k)
# 2*2*5 = len(n)
k.ravel() # gives u the single dimension with all the elements


# In[ ]:


print(np.random.rand())
print(np.random.rand()*100)


# In[ ]:


print(np.random.randint(100))
print(np.random.randint(10,30))


# In[ ]:


np.random.randint(10,high=30,size=15).reshape(5,3)


# In[ ]:


print(np.floor(3.99))
print(np.ceil(3.09))


# In[ ]:


print(np.round(1.0))
print(np.round(1.1))
print(np.round(1.2))
print(np.round(1.3))
print(np.round(1.4))
print(np.round(1.5))
print(np.round(1.6))
print(np.round(1.7))
print(np.round(1.8))
print(np.round(1.9))


# In[ ]:


print(np.round(0.5))
print(np.round(1.5))
print(np.round(2.5))
print(np.round(3.5))
print(np.round(4.5))
print(np.round(5.5))
print(np.round(6.5))
print(np.round(7.5))
print(np.round(8.5))
print(np.round(9.5))


# In[ ]:


o=np.linspace(0,30,4) # Equally distributes
print(o)
p=np.linspace(0,20,4) # Equally distributes
print(p)
q=np.linspace(0,10,4) # Equally distributes
print(q)


# In[ ]:


r=np.array([2,3,4,5])
s=r # refering the same data with a different name
t= r.copy() # copying the data to a different variable.


# **A 'n' dimensional array is just a collection of (n-1) dimensional array.**
