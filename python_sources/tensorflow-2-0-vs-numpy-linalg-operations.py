#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import scipy as sp


# In[ ]:


get_ipython().run_cell_magic('time', '', 'm1 = np.random.randn(5000, 5000)\nm2 = np.random.randn(5000, 5000)\nm3 = np.random.randn(10000, 10000)\nv1 = np.random.randn(5000, 1)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 't1 = tf.convert_to_tensor(m1)\nt2 = tf.convert_to_tensor(m2)\nt3 = tf.convert_to_tensor(m3)\ntv1 = tf.convert_to_tensor(v1)')


# In[ ]:


get_ipython().run_line_magic('time', 'tf.matmul(t1, t2)')
get_ipython().run_line_magic('time', 'np.matmul(m1, m2)')

print()


# In[ ]:


get_ipython().run_line_magic('time', 'tf.linalg.inv(t1)')
get_ipython().run_line_magic('time', 'np.linalg.inv(m1)')

print()


# In[ ]:


get_ipython().run_line_magic('timeit', 'tf.linalg.norm(t3)')
get_ipython().run_line_magic('timeit', 'np.linalg.norm(m3)')
print()


# In[ ]:


get_ipython().run_line_magic('time', 'tf.linalg.svd(t1)')
get_ipython().run_line_magic('time', 'np.linalg.svd(m1)')
print()


# In[ ]:


get_ipython().run_line_magic('time', 'tf.linalg.det(t1)')
get_ipython().run_line_magic('time', 'np.linalg.det(m1)')
print()


# In[ ]:


get_ipython().run_line_magic('timeit', 'tf.transpose(t3)')
get_ipython().run_line_magic('timeit', 'np.transpose(m3)')
print()


# In[ ]:


get_ipython().run_line_magic('time', 'tf.linalg.lstsq(t1, tv1, fast=False)')
get_ipython().run_line_magic('time', 'np.linalg.lstsq(m1, v1)')
print()


# In[ ]:


get_ipython().run_line_magic('time', 'tf.linalg.eigvalsh(t1)')
get_ipython().run_line_magic('time', 'np.linalg.eigvalsh(m1)')
print()


# In[ ]:


get_ipython().run_line_magic('time', 'tf.linalg.eigh(t1)')
get_ipython().run_line_magic('time', 'np.linalg.eigh(m1)')
print()


# In[ ]:


get_ipython().run_line_magic('time', 'tf.linalg.lu(t1)')
get_ipython().run_line_magic('time', 'sp.linalg.lu(m1)')
print()


# In[ ]:


get_ipython().run_line_magic('time', 'tf.linalg.qr(t1)')
get_ipython().run_line_magic('time', 'np.linalg.qr(m1)')
print()

