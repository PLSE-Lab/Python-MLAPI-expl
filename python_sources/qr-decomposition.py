#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error


# **If**** it is a linear algebra problem then we can use matrix math to solve it
# Ax = b
# Unfortunately we have more unknowns (columns) than equations (rows).  We say A is a fat matrix which as we know in the 21st century that fat is bad
# However we can transpose A so that it is nice and thin and then do QR decomposition of Atranspose

# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


A = train[train.columns[2:]].values
b = train.target.values


# In[ ]:


Q,R = np.linalg.qr(A.T)


# Now for the 2 norm all we need to solve is
# RTransposeU = b where U is the unknown

# In[ ]:


RTinv = np.linalg.pinv(R.T)


# In[ ]:


U = np.dot(RTinv,b)


# In[ ]:


predictions = np.dot(R.T,U).clip(train.target.min(),train.target.max()) #We have no conditions so just keep the values in the range of min and max for target


# In[ ]:


np.sqrt(mean_squared_error(np.log1p(b),np.log1p(predictions)))

