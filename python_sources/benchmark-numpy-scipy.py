#!/usr/bin/env python
# coding: utf-8

# In[9]:


import sys
import timeit

import numpy as np
import scipy as sp
from numpy.distutils.system_info import get_info

print("maxint:  %i\n" % sys.maxsize)


# In[3]:


info = get_info('blas_opt')
print('BLAS info:')

for kk, vv in info.items():
    print(' * ' + kk + ' ' + str(vv))


# In[ ]:


print('numpy version: {}'.format(numpy.__version__))
print(numpy.show_config())


# ## scipy config

# In[8]:


print('scipy version: {}'.format(scipy.__version__))
print(scipy.show_config())


# # Benchmark dotproduct
# reference:
# 
# 1. https://scipy.github.io/old-wiki/pages/PerformanceTips

# In[26]:


N = int(1e6)
n = 40
A = np.ones((N,n))
C = np.dot(A.T, A)

AT_F = np.ones((n,N), order='F')
AT_C = np.ones((n,N), order='C')


# In[32]:


#numpy.dot
print('')
get_ipython().run_line_magic('timeit', 'np.dot(A.T, A)  #')

print('')
get_ipython().run_line_magic('timeit', 'np.dot(AT_F, A)  #')

print('')
get_ipython().run_line_magic('timeit', 'np.dot(AT_C, A)  #')


# In[34]:


import scipy.linalg.blas
get_ipython().run_line_magic('timeit', 'scipy.linalg.blas.dgemm(alpha=1.0, a=A.T, b=A.T, trans_b=True)')

get_ipython().run_line_magic('timeit', 'scipy.linalg.blas.dgemm(alpha=1.0, a=A, b=A, trans_a=True)')


# # Benchmark pairwaise distance
# reference:
# 1. https://jakevdp.github.io/blog/2013/06/15/numba-vs-cython-take-2/

# In[10]:


X = np.random.random((1000, 3))


# In[11]:


# Numpy Function With Broadcasting
def pairwise_numpy(X):
    return np.sqrt(((X[:, None, :] - X) ** 2).sum(-1))
get_ipython().run_line_magic('timeit', 'pairwise_numpy(X)')


# In[12]:


# Pure python function
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D
get_ipython().run_line_magic('timeit', 'pairwise_python(X)')


# In[13]:


# Numba wrapper
from numba import double
from numba.decorators import jit, autojit

pairwise_numba = autojit(pairwise_python)

get_ipython().run_line_magic('timeit', 'pairwise_numba(X)')


# In[17]:


#optimize cython function
get_ipython().run_line_magic('load_ext', 'Cython')


# In[18]:


get_ipython().run_cell_magic('cython', '', '\nimport numpy as np\ncimport cython\nfrom libc.math cimport sqrt\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef pairwise_cython(double[:, ::1] X):\n    cdef int M = X.shape[0]\n    cdef int N = X.shape[1]\n    cdef double tmp, d\n    cdef double[:, ::1] D = np.empty((M, M), dtype=np.float64)\n    for i in range(M):\n        for j in range(M):\n            d = 0.0\n            for k in range(N):\n                tmp = X[i, k] - X[j, k]\n                d += tmp * tmp\n            D[i, j] = sqrt(d)\n    return np.asarray(D)')


# In[19]:


get_ipython().run_line_magic('timeit', 'pairwise_cython(X)')


# In[20]:


# scipy pairwise distance
from scipy.spatial.distance import cdist
get_ipython().run_line_magic('timeit', 'cdist(X, X)')


# In[21]:


from sklearn.metrics import euclidean_distances
get_ipython().run_line_magic('timeit', 'euclidean_distances(X, X)')


# In[ ]:




