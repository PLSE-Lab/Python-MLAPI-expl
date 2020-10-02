#!/usr/bin/env python
# coding: utf-8

# # <i>On demand recomputing (disk-caching)</i> using [Joblib](https://joblib.readthedocs.io/en/latest/index.html)

# ### Install Joblib package

# In[ ]:


get_ipython().system('pip install joblib')


# ### Import the Memory class

# In[ ]:


from joblib import Memory


# ### Import other modules

# In[ ]:


import numpy as np
from time import sleep 


# ### Ignore any warnings raised by Jupyter notebook

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# ### Let's create our cache directory

# In[ ]:


pwd = "/kaggle/working/"
cache_dir = pwd + 'cache_dir'
mem = Memory(cache_dir)


# #### Directory with name '/kaggle/working/cache_dir/' has been created

# In[ ]:


get_ipython().system('ls -ld $pwd*/')


# ### Define some large inputs

# In[ ]:


input1 = np.vander(np.arange(10**4)).astype(np.float)
input2 = np.vander(np.random.uniform(low=0,high=10**5, size=5000))
print("Shape of input1: ",input1.shape)
print("Shape of input2: ",input2.shape)


# <br>

# ## There are two ways to pass a function to Memory.cache

# ### Method 1: Passing a function to Memory.cache

# #### Define function

# In[ ]:


def func(x):
    print("Example of Computationally intensive function!")
    print("The result is not cached for this particular input")
    sleep(4.0)
    return np.square(x)


# #### Pass it to Memory.cache function

# In[ ]:


func_mem = mem.cache(func, verbose=0)


# #### Before we begin, let's check the cache directory size

# In[ ]:


get_ipython().system('du -sh $cache_dir')


# #### Let's checkout some caching results

# In[ ]:


get_ipython().run_cell_magic('time', '', 'input1_result = func_mem(input1)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'input1_cache_result = func_mem(input1)')


# #### Check the time difference in execution. When we fetch the results of <i>func_mem</i> with same parameters i.e. input1, we use the <span style="color:red">cached results instead of doing the computations again</span>. 
# 
# <i><u>Note</u>: The Memory.cache only caches the result returned by func_mem. Print statement result is not printed.</i>
# 
# #### Memory class uses fast cryptographic hashing of the input arguments to check if they have been computed

# 
# 

# #### The result for input2 hasn't been cached

# In[ ]:


get_ipython().run_cell_magic('time', '', 'input2_result = func_mem(input2)')


# #### Notice the time difference in execution for the above code execution for input2

# In[ ]:


get_ipython().run_cell_magic('time', '', 'input2_cache_result = func_mem(input2)')


# <br>

# #### Let's again check the cache directory size.

# In[ ]:


get_ipython().system('du -sh $cache_dir')


# #### *We see that there is change in size*

# <br>

# ### Method 2: Memory.cache as a decorator

# In[ ]:


@mem.cache(verbose=0)
def func_as_decorator(x):
    print("Example of Computationally intensive function!")
    print("The result is not cached for this particular input")
    sleep(4.0)
    return np.square(x)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'input1_decorator_result = func_as_decorator(input1)')


# #### Notice the time difference in execution

# In[ ]:


get_ipython().run_cell_magic('time', '', 'input1_decorator_result = func_as_decorator(input1)')


# <br>

# ## Using Memmapping (memory mapping) if working with numpy

# ### Memmapping speeds up cache looking when reloading large numpy arrays

# In[ ]:


cache_dir2 = pwd + 'cache_dir2'
memory2 = Memory(cache_dir2, mmap_mode='c')


# In[ ]:


@memory2.cache(verbose=0)
def func_memmap(x):
    print("Example of Computationally intensive function!")
    print("The result is not cached for this particular input")
    sleep(4.0)
    return np.square(x)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'input1_memmap = func_memmap(x=input1)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'input1_memmap = func_memmap(x=input1)')


# [Check the time difference in execution when using memory map vs non memory map](#Notice-the-time-difference-in-execution)

# <br>

# ## Clearning cache

# ### Clear function's cache

# In[ ]:


# Disk utilization before clearning function cache
get_ipython().system('du -sh $cache_dir')


# In[ ]:


func_mem.clear()
func_as_decorator.clear()


# In[ ]:


# Disk utilization after clearning function cache
get_ipython().system('du -sh $cache_dir')


# #### Notice above the disk utilization of "*/kaggle/working/cache_dir*" before and after clearing function cache

# ### Erase complete cache directory

# In[ ]:


mem.clear()


# #### Let's check if the cache directory has been cleared

# In[ ]:


get_ipython().system('du -sh $cache_dir')


# ## Congratulations on completing disk-caching using Joblib

# ## Looking forward for your feedback in the comments section below
# ### If you liked this kernel please hit the Upvote button.

# # Next - Learn how to parallelize `for loops` using [Joblib](https://joblib.readthedocs.io/en/latest/index.html) in the most easiest way: https://www.kaggle.com/karanpathak/parallelize-loops-using-joblib

# In[ ]:




