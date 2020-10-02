#!/usr/bin/env python
# coding: utf-8

# # Profiling Your GPU Runtime
# ## By Jeff Hale
# 
# See this Medium article for accompanying discussion comparing Kaggle and Colab GPU environments.

# First let's look at some GPU specs.
# ## GPU

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


get_ipython().system('cat /usr/local/cuda/version.txt')


# Now let's look at CPU info.
# ## CPU

# In[ ]:


get_ipython().system('cat /proc/cpuinfo')


# In[ ]:


import multiprocessing
multiprocessing.cpu_count()


# ## Memory

# In[ ]:


get_ipython().system('cat /proc/meminfo')


# ## Disk Space

# In[ ]:


get_ipython().system('df -h ')


# I hope this notebook helps you find your runtime specs faster. Check out the accompanying Medium article for a larger discussion.
