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


np.arange(0, 10)


# In[ ]:


np.arange(0, 20, 2)


# In[ ]:


np.zeros((5, 5))


# In[ ]:


np.ones((2, 4), dtype = float)


# In[ ]:


array = np.random.randint(0, 10, 5)


# In[ ]:


array.max()


# In[ ]:


array.mean()


# In[ ]:


array


# In[ ]:


array2 = np.random.randint(0, 100, 10)


# In[ ]:


array2


# In[ ]:


array2.reshape((2, 5))


# In[ ]:


array3 = np.arange(0, 16, dtype = float).reshape((4, 4))


# In[ ]:


array3


# In[ ]:


A = np.array([[2, 5], [3, 8]])


# In[ ]:


A


# In[ ]:


B = np.array([[3, 4], [2, 3]])


# In[ ]:


B


# In[ ]:


A + B


# In[ ]:


A * B


# In[ ]:


A


# In[ ]:


B


# In[ ]:


A.dot(B)


# In[ ]:


A.sum()


# In[ ]:


B.sum()


# In[ ]:


A.max()


# In[ ]:


B.max()


# In[ ]:


A


# In[ ]:


A.sum(axis=0)


# In[ ]:


A.sum(axis=1)


# In[ ]:


A.min(axis=0)


# In[ ]:


A.T


# # Assignment (02.04.2019)

# In[ ]:


# Create array of 10 zeroes
# Create array of 10 ones
# Create array of integers from 10 to 50
# Create 3 * 4 matrix with values 0 to 12
# Interview question: Diff between numpy and list ( Why should we use numpy arange instead of python list?)


# In[ ]:


np.zeros((10, 10))


# In[ ]:


np.ones((10, 10))


# In[ ]:


np.arange(10, 50)


# In[ ]:


np.arange(0, 12).reshape((3 ,4))


# ### Diff between numpy and list ( Why should we use numpy arange instead of python list?)
# #### Size - Numpy data structures take up less space
# #### Performance - they have a need for speed and are faster than lists
# #### Functionality - SciPy and NumPy have optimized functions such as linear algebra operations built in.

# In[ ]:




