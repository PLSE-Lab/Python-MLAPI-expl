#!/usr/bin/env python
# coding: utf-8

# # Newell-Littlewood numbers
# https://arxiv.org/abs/2005.09012

# In[ ]:


# We need CMake 3.11+ and Python3 (with dev libraries).
get_ipython().system('pip3 install cmake==3.17.3')
get_ipython().system('cmake --version')

# The current version of `nlnum` is 0.0.5. This may change with time.
# You can remove the `==0.0.5` to fetch the latest version.
get_ipython().system('pip3 install nlnum==0.0.5')


# In[ ]:


import numpy as np
from nlnum import lrcoef, nlcoef, nlcoef_slow


# The below code defines the function $F(t) = N_{t\mu,t\nu,t\lambda}$.

# In[ ]:


N = lambda mu, nu, lam: lambda t: nlcoef(t*mu, t*nu, t*lam)


# Let $\mu = \nu = \lambda = (2, 1, 1)$.

# In[ ]:


# Here are the input partitions.
mu = nu = lam = np.array([2, 1, 1])

# This is the scaling function F(t).
F = N(mu, nu, lam)


# The below code computes $F(0),\ldots, F(13)$ using the definition (1.1) of $N_{\mu,\nu,\lambda}$ **without** parallelization.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'np.array([ F(t) for t in range(13+1) ])')


# The below code computes $F(6)$ using Proposition 2.3 **without** parallelization.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'nlcoef_slow(6*mu, 6*nu, 6*lam)')

