#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


a = np.zeros(3)
a


# In[ ]:


type(a)


# In[ ]:


a = np.zeros(3)
type(a[0])


# In[ ]:


a = np.zeros(3, dtype=int)
type(a[0])


# In[ ]:


z = np.zeros(10)


# In[ ]:


z.shape


# In[ ]:


z.shape = (10, 1)
z


# In[ ]:


z = np.zeros(4)
z.shape = (2, 2)
z


# In[ ]:


z = np.empty(3)
z


# In[ ]:


z = np.linspace(2, 4, 5)  # From 2 to 4, with 5 elements


# In[ ]:


z = np.identity(2)
z


# In[ ]:


z = np.array([10, 20])                 # ndarray from Python list
z


# In[ ]:


type(z)


# In[ ]:


z = np.array((10, 20), dtype=float)    # Here 'float' is equivalent to 'np.float64'
z


# In[ ]:


z = np.array([[1, 2], [3, 4]])         # 2D array from a list of lists
z


# In[ ]:


na = np.linspace(10, 20, 2)
na is np.asarray(na)   # Does not copy NumPy arrays


# In[ ]:


na is np.array(na)  


# In[ ]:


z = np.linspace(1, 2, 5)
z


# In[ ]:


z[0]


# In[ ]:


z[0:2]  # Two elements, starting at element 0


# In[ ]:


z[-1]


# In[ ]:


z = np.array([[1, 2], [3, 4]])
z


# In[ ]:


z[0, 0]


# In[ ]:


z[0, 1]


# In[ ]:


z[0, :]


# In[ ]:


z[:, 1]


# In[ ]:


z = np.linspace(2, 4, 5)
z


# In[ ]:


indices = np.array((0, 2, 3))
z[indices]


# In[ ]:


z


# In[ ]:


d = np.array([0, 1, 1, 0, 0], dtype=bool)
d


# In[ ]:


z[d]


# In[ ]:


z = np.empty(3)
z


# In[ ]:


z[:] = 42
z


# In[ ]:


z = np.array([1, 2, 3])
np.sin(z)


# In[ ]:


n = len(z)
y = np.empty(n)
for i in range(n):
    y[i] = np.sin(z[i])


# In[ ]:


z


# In[ ]:


(1 / np.sqrt(2 * np.pi)) * np.exp(- 0.5 * z**2)


# In[ ]:


def f(x):
    return 1 if x > 0 else 0


# In[ ]:


x = np.random.randn(4)
x


# In[ ]:


np.where(x > 0, 1, 0)  # Insert 1 if x > 0 true, otherwise 0


# In[ ]:


f = np.vectorize(f)
f(x)                # Passing the same vector x as in the previous example


# In[ ]:


z = np.array([2, 3])
y = np.array([2, 3])
z == y


# In[ ]:


y[0] = 5
z == y


# In[ ]:


z != y

