#!/usr/bin/env python
# coding: utf-8

# **Numpy Array indexing and Selection**

# In[ ]:


import numpy as np 
arr = np.arange(0,11)
arr


# In[ ]:


arr[4]


# In[ ]:


arr[2:4]


# In[ ]:


arr[5:]


# **choosing position and updating the values **

# In[ ]:


slice_arr = np.arange(0,7)
slice_arr


# In[ ]:


slice_arr[0:4] =1001
slice_arr


# In[ ]:


slice_arr[:]=77
slice_arr


# **copy() method**

# In[ ]:


arr_copy = arr.copy()
arr_copy


# **Indexing 2d array or Matrices**

# In[ ]:


arr_2d =[[5,10,15],[20,25,30],[35,40,45],[50,55,60]]
arr_2d


# In[ ]:


arr_2dm = np.array(arr_2d)
arr_2dm


# In[ ]:


arr_2dm[1]


# In[ ]:


arr_2dm[:3,2:]


# In[ ]:


arr_2dm[:3]


# In[ ]:


arr_2dm[:2,2:]


# In[ ]:


arr_2dm[:1,2]


# In[ ]:


arr_2d[3][2]


# Getting Boolean true or false array by comparing numeric array

# In[ ]:


arr = np.arange(11)
arr


# In[ ]:


bool_arr = arr>5
bool_arr


# **Getting elements of true by comparing**

# In[ ]:


arr[bool_arr]


# In[ ]:


arr[arr<3]


# In[ ]:


arr_2 = np.arange(50).reshape(5,10)
arr_2


# In[ ]:


arr_2[1:3,6:8]


# 

# **NUMPY OPERATIONS**

# In[ ]:


import numpy as np 
ar = np.arange(0,11)
ar +ar


# In[ ]:


ar -ar


# In[ ]:


ar*ar


# In[ ]:


ar+100


# In[ ]:


np.sqrt(ar)


# In[ ]:


np.exp(ar)


# In[ ]:


ar**2


# In[ ]:


np.sin(ar)


# In[ ]:


ar.max


# **NUMPY OPERATIONS**

# In[ ]:


import numpy as np


# In[ ]:


arr = np.zeros(10)
arr


# In[ ]:


arr = np.ones(10)
arr


# In[ ]:


arr =np.eye(5)
arr


# In[ ]:


arr = np.arange(0,11,5)
arr


# In[ ]:


arr = np.ones(10)*5
arr


# In[ ]:


arr = np.arange(10,51)
arr


# In[ ]:


arr = np.arange(10,51,2)
arr


# In[ ]:


arr = np.arange(0,9)
arr


# In[ ]:


arr = np.array(arr)
type(arr)
arr.reshape(3,3)


# In[ ]:


arr = np.eye(3)
arr


# In[ ]:


np.random.rand(1)


# In[ ]:


np.random.randn(25)


# In[ ]:


np.linspace(0,1,100)


# In[ ]:


np.linspace(0,1,20)


# In[ ]:


mat = np.arange(1,26).reshape(5,5)
mat


# In[ ]:


mat[3:]


# In[ ]:


mat[2:]


# In[ ]:


mat[2:,1:5]


# In[ ]:


mat[:3,1:2]


# In[ ]:


mat[3][4]


# In[ ]:


np.sum(mat)


# In[ ]:


np.std(mat)


# In[ ]:


mat.sum(axis=0)

