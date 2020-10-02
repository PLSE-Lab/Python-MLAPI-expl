#!/usr/bin/env python
# coding: utf-8

# # PYTHON NUMPY
# This is my third kernel. If you like it, Please upvote! 
# I prepared this kernel while i was studying Udemy course. 
# https://www.udemy.com/course/veri-bilimi-ve-makine-ogrenmesi-icin-python/
# 
# **SECTIONS**
# # CREATE NUMPY ARRAY
# 
# # NUMPY INDEXING
# 
# # NUMPY TRANSACTIONS

# # CREATE NUMPY ARRAY

# In[ ]:


import numpy as np


# In[ ]:


x = np.array([1,2,3,4])


# In[ ]:


type(x)


# In[ ]:


np.zeros((3,4))


# In[ ]:


np.ones((2,4))


# In[ ]:


np.ones((2,4))*5


# In[ ]:


n = np.arange(10,50,5)
n


# In[ ]:


decimal = np.arange(0,2,0.2)
decimal


# In[ ]:


lin = np.linspace(0,4,12)
lin


# In[ ]:


rast = np.random.rand(3,4)
rast


# In[ ]:


n = np.random.randn(2,5)*5
n


# In[ ]:


np.eye(5)


# In[ ]:


c= np.random.randint(1,50,15)
c


# In[ ]:


y = np.arange(24)
y


# In[ ]:


y.reshape(6,4)


# In[ ]:


y.reshape(2,3,4)


# In[ ]:


c.max()


# In[ ]:


c.min()


# In[ ]:


c.argmax()


# In[ ]:


c.argmin()


# In[ ]:


a = np.arange(50)
a.shape = 2,-1,5
a.shape


# In[ ]:


a


# # NUMPY INDEXING

# # One-Dimensional Array

# In[ ]:


a = np.arange(10)**2


# In[ ]:


a


# In[ ]:


a[3]


# In[ ]:


a[3:6]


# In[ ]:


a[4] = 1000
a


# In[ ]:


for i in a:
    print(i*2)


# # Multidimensional Array

# In[ ]:


y = np.random.random(16)


# In[ ]:


y


# In[ ]:


y = y.reshape(4,4)
y


# In[ ]:


y.ravel()


# In[ ]:


y[0:2,1]


# In[ ]:


y[:,3]


# In[ ]:


for row in y:
    print(row)


# In[ ]:


for row in y.flat:
    print(row)


# In[ ]:


for row in np.ndenumerate(y):
    print(row)


# In[ ]:


a = np.array([2,3])
b = np.array([5,6])


# In[ ]:


np.hstack((a,b))


# In[ ]:


np.vstack((a,b))


# In[ ]:


a = np.array([[1,2],[3,4]])
b = np.array([[5,6]])
np.concatenate((a,b),axis = 0)


# In[ ]:


np.concatenate((a,b.T),axis=1)


# # NUMPY TRANSACTIONS

# In[ ]:


a = np.array([20,40,60,80])
a


# In[ ]:


b= np.arange(4)
b


# In[ ]:


c = a+b
c


# In[ ]:


a**2


# In[ ]:


a>50


# In[ ]:


a += 10


# In[ ]:


a


# In[ ]:


x = np.random.random(12)
x


# In[ ]:


x.max()


# In[ ]:


x.min()


# In[ ]:


x.sum()


# In[ ]:


b = np.arange(15).reshape(3,5)
b


# In[ ]:


b.sum(axis = 0)


# In[ ]:


b.sum(axis = 1)


# In[ ]:


b.max(axis = 1)


# In[ ]:


b.cumsum(axis=1)


# In[ ]:


k = np.random.random(10)


# In[ ]:


k


# In[ ]:


k.mean()


# In[ ]:


np.median(k)


# In[ ]:


np.std(k)


# In[ ]:


np.var(k)


# In[ ]:


copied = k.copy()


# In[ ]:


copied


# In[ ]:


city = np.array(["ankara","istanbul","bursa","ankara","izmir","bursa","izmir"])
np.unique(city)


# In[ ]:




