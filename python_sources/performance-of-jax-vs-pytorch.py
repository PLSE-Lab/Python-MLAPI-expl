#!/usr/bin/env python
# coding: utf-8

# # Performance of JAX vs PyTorch
# 
# Let's compare how fast two libraries can calculate a gradient of the same function: JAX vs PyTorch. No hardware acceleration will be enabled, we will use just CPU (GPU is disabled in this notebook).

# ## JAX

# Install and import JAX, enable usage of 64-bit floats and CPU for computations.

# In[ ]:


get_ipython().system('pip -q install jax jaxlib')

get_ipython().run_line_magic('env', 'JAX_ENABLE_X64=1')
get_ipython().run_line_magic('env', 'JAX_PLATFORM_NAME=cpu')

import jax.numpy as np
from jax import grad, ops, jit, lax


# Next we will define a toy function which will be used for our tests. It receives an array as input, performs some computations on it and returns a scalar. The problem is O(n) hard - the longer the input array, the linearly longer it takes to calculate the result.
# 
# As gnecula and mattjj kindly [explained](https://github.com/google/jax/issues/1832), it's better to use lax.scan in this case instead of a `for` loop, so it is commented out.

# In[ ]:


def func_jax(x):
    t = len(x)
    f = np.zeros(t)
    
    #for i in range(1, t):
    #    f = ops.index_update(f, i, x[i]+f[i-1])
    
    f = lax.scan(lambda f, i: (ops.index_update(f, i, x[i] + f[i-1]), None), f, np.arange(1, t))
    
    return np.sum(f[0])


# Perform a sanity check.

# In[ ]:


func_jax(np.ones(100))


# Measure performance of a gradient calculation for different array length. Run only one loop to exclude any caching if it exists. Also add a `.block_until_ready()` so we are not just timing dispatch time (due to asynchronous dispatch).

# In[ ]:


get_ipython().run_line_magic('timeit', 'grad(func_jax)(np.ones(10)).block_until_ready()')


# In[ ]:


get_ipython().run_line_magic('timeit', 'grad(func_jax)(np.ones(100)).block_until_ready()')


# In[ ]:


get_ipython().run_line_magic('timeit', 'grad(func_jax)(np.ones(1000)).block_until_ready()')


# In[ ]:


get_ipython().run_line_magic('timeit', 'grad(func_jax)(np.ones(10000)).block_until_ready()')


# Jitted versions have almost the same timings. 

# In[ ]:


get_ipython().run_line_magic('timeit', 'jit(grad(func_jax))(np.ones(10)).block_until_ready()')


# In[ ]:


get_ipython().run_line_magic('timeit', 'jit(grad(func_jax))(np.ones(100)).block_until_ready()')


# In[ ]:


get_ipython().run_line_magic('timeit', 'jit(grad(func_jax))(np.ones(1000)).block_until_ready()')


# In[ ]:


get_ipython().run_line_magic('timeit', 'jit(grad(func_jax))(np.ones(10000)).block_until_ready()')


# Looks like the problem is O(1) now!

# ## PyTorch

# Let's do the same with PyTorch.

# In[ ]:


import torch


# A test function is still the same with a slight cosmetic modifications.

# In[ ]:


def func_torch(x):
    t = len(x)
    f = torch.zeros(t)
    for i in range(1, t):
        f[i] = x[i] + f[i-1]
    return f.sum()


# A sanity check again. Just a quick test to check that we implemented the same function.

# In[ ]:


func_torch(torch.ones(100, dtype=torch.float64))


# Define a gradient function.

# In[ ]:


def grad_torch(length):
    x = torch.ones(length, requires_grad=True, dtype=torch.float64)
    func_torch(x).backward()
    return x.grad


# Now measure performance.

# In[ ]:


get_ipython().run_line_magic('timeit', 'grad_torch(10)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'grad_torch(100)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'grad_torch(1000)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'grad_torch(10000)')


# PyTorch is fast, but looks like it solves the problem in O(n) time.
