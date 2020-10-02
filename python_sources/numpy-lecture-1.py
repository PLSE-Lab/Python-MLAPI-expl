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


# In[ ]:


import numpy as np


# In[76]:


np.__version__


# In[ ]:


x = np.array([1, 2, 10])


# In[ ]:


x


# In[ ]:


x = np.array([1, 2.5, 10])
x


# In[ ]:


x = np.array([1, 2.5, 'Masha'])
x


# In[ ]:


get_ipython().set_next_input('x = np.random.randint');get_ipython().run_line_magic('pinfo', 'np.random.randint')


# In[ ]:


x = np.random.randint(-10, 10, 20)
x


# In[ ]:


x = np.random.normal(175, 7, 20)


# # x

# In[ ]:


y = np.random.uniform(168, 182, 20)
y


# In[ ]:


x = np.zeros(10)
x


# In[ ]:


A = np.zeros((10, 10))
A


# In[ ]:


A = np.random.normal(0, 5, (10, 10))
A


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'sum([i for i in range(1000000)])')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'np.arange(1000000).sum()')


# In[ ]:


x = np.random.uniform(0, 10, 5)
y = np.arange(5, 10)
A = np.random.normal(5, 2.5, (5, 5))


# In[ ]:


x.size


# In[ ]:


A.size


# In[ ]:


x.ndim


# In[ ]:


A.ndim


# In[ ]:


A.shape


# In[ ]:


A.shape[0]


# In[ ]:


x.shape


# In[ ]:


x = x[np.newaxis, :]
x.shape


# In[ ]:


y = y[:, np.newaxis]
y.shape


# In[ ]:


x = x.reshape((5,))


# In[ ]:


x[0]


# In[ ]:


x[:3]


# In[ ]:


x[-2:]


# In[ ]:


A[0, 0]


# In[ ]:


A[0, 0] = -100


# In[ ]:


A


# In[ ]:


A[1]


# In[ ]:


A[1, :] = x


# In[ ]:


A


# In[ ]:


A[:, 3]


# In[ ]:


A[:2, :2]


# In[ ]:


np.linalg.det(A)


# In[ ]:


np.linalg.matrix_rank(A)


# In[48]:


A[[0, 2], :]


# In[49]:


x


# In[50]:


mean_x = x.mean()


# In[51]:


x[x > mean_x]


# In[52]:


x > 0


# In[54]:


x > mean_x


# In[56]:


x[[False, False, True, True, False]]


# In[57]:


A


# In[59]:


A = np.arange(9).reshape((3, 3))
A


# In[60]:


A[A[:, 0]>0]


# In[61]:


A[A[:, 0]>0, :]


# In[62]:


A[:, A[0, :]>0]


# In[63]:


print(x)
print(y)


# In[64]:


y = y.reshape((5,))
print(x, y)


# In[65]:


x + y


# In[66]:


x + 1


# In[67]:


1 / x


# In[69]:


np.sin(x)


# In[70]:


x * y


# In[71]:


x.dot(y)


# In[74]:


A.dot(x[:3])


# In[75]:


A.T


# In[ ]:




