#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# https://www.kaggle.com/dromosys/mnst-fastai

# In[ ]:


a = np.array([10, 6, -4])
b = np.array([2, 8, 7])
a,b


# In[ ]:


a + b


# In[ ]:


a > 0


# In[ ]:


a + 1


# In[ ]:


m = np.array([[1, 2, 3], [4,5,6], [7,8,9]]); m


# In[ ]:


2*m


# In[ ]:


c = np.array([10,20,30]); c


# In[ ]:


m + c


# In[ ]:


np.broadcast_to(c[:,None], m.shape)


# In[ ]:


np.broadcast_to(np.expand_dims(c,0), (3,3))


# In[ ]:


c.shape


# In[ ]:


np.expand_dims(c,0).shape


# In[ ]:


m + np.expand_dims(c,0)


# In[ ]:


np.expand_dims(c,1)


# In[ ]:


c[:, None].shape


# In[ ]:


m + np.expand_dims(c,1)


# In[ ]:


np.broadcast_to(np.expand_dims(c,1), (3,3))


# In[ ]:


m, c


# In[ ]:


m @ c


# In[ ]:


xg,yg = np.ogrid[0:5, 0:5]; xg,yg


# In[ ]:


xg+yg


# In[ ]:


m,c


# In[ ]:


m * c


# In[ ]:


(m * c).sum(axis=1)


# The website matrixmultiplication.xyz provides a nice visualization of matrix multiplcation http://matrixmultiplication.xyz/

# In[ ]:


n = np.array([[10,40],[20,0],[30,-5]]); n


# In[ ]:


m @ n


# In[ ]:


(m * n[:,0]).sum(axis=1)


# In[ ]:


(m * n[:,1]).sum(axis=1)


# In[ ]:





# In[ ]:




