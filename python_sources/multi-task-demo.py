#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# prepare
import os
import multiprocessing.pool
import requests
import random

# use multiple sites because I am not doing a DDoS.
sites = [
    'http://www.ibm.com',
    'http://www.google.com',
    'http://www.facebook.com',
    'http://www.youtube.com',
    'http://www.microsoft.com',
    'http://www.zhihu.com',
    'http://www.github.com',
    'http://www.baidu.com',
    'http://www.qq.com',
    'http://www.taobao.com',
    'http://www.amazon.com',
    'http://www.kaggle.com',
    'http://www.bilibili.com',
    'http://www.bing.com',
]

def cpu_busy(n):
    for _ in range(10000000):
        n * n
    return n * n

def io_busy(n):
    r = requests.get(random.choice(sites))
    return n * n

N = 16
f'CPU count={os.cpu_count()}'


# In[ ]:


get_ipython().run_cell_magic('time', '', '[io_busy(x) for x in range(N)]')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with multiprocessing.pool.ThreadPool(os.cpu_count()) as pool:\n    result = pool.map(io_busy, range(N))\nresult')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with multiprocessing.pool.Pool(os.cpu_count()) as pool:\n    result = pool.map(io_busy, range(N))\nresult')


# In[ ]:


get_ipython().run_cell_magic('time', '', '[cpu_busy(x) for x in range(N)]')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with multiprocessing.pool.ThreadPool(os.cpu_count()) as pool:\n    result = pool.map(cpu_busy, range(N))\nresult')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'with multiprocessing.pool.Pool(os.cpu_count()) as pool:\n    result = pool.map(cpu_busy, range(N))\nresult')


# In[ ]:




