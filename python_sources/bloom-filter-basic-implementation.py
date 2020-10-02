#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install git+https://github.com/jaybaird/python-bloomfilter.git')


# In[ ]:


from pybloom import BloomFilter
f = BloomFilter(capacity=1000, error_rate=0.001)


# In[ ]:


f.add("One String")


# In[ ]:


"One String" in f


# In[ ]:


"1 String" in f


# In[ ]:




