#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


pip install torch


# In[ ]:


import torch
import numpy as np


# In[ ]:


a=np.array(1)
b=torch.tensor(1)
print(a)
print(b)


# In[ ]:


type(a)


# In[ ]:


type(b)


# In[ ]:


c=np.array(2)
d=np.array(1)
print(c,d)


# In[ ]:


print(c-d)


# In[ ]:


print(c+d)


# In[ ]:


print(c/d)


# In[ ]:


a=torch.tensor(4)
b=torch.tensor(2)
print(a/b)


# In[ ]:


print(a+b)


# In[ ]:


a=np.zeros((3,3))
print(a)


# In[ ]:


a.ndim


# In[ ]:


a.size


# In[ ]:


a.shape


# In[ ]:


a=np.zeros((4,4))
a


# In[ ]:


np.random.seed(42)
a=np.random.randn(3,3)
a


# In[ ]:


torch.manual_seed(42)
a=torch.randn(3,3)
a


# In[ ]:


np.random.seed(42)
a=np.random.randn(3,3)
b=np.random.randn(3,3)


# In[ ]:


print(np.add(a,b))


# In[ ]:


print(np.subtract(a,b))


# In[ ]:


print(np.dot(a,b))


# In[ ]:


print(np.divide(a,b))


# In[ ]:


print(a)


# In[ ]:


print(np.transpose(a))


# In[ ]:


a=np.array([[1,2],[3,4]])
print(a)


# In[ ]:


a.shape

