#!/usr/bin/env python
# coding: utf-8

# In[1]:


#GPU count and name
get_ipython().system('nvidia-smi -L')


# In[3]:


get_ipython().system("lscpu |grep 'Model name'")


# In[4]:


#no.of sockets i.e available slots for physical processors
get_ipython().system("lscpu | grep 'Socket(s):'")


# In[5]:


#no.of cores each processor is having
get_ipython().system("lscpu | grep 'Core(s) per socket'")


# In[6]:


#no.of threads each core is having
get_ipython().system("lscpu | grep 'Thread(s) per core'")


# In[7]:


get_ipython().system("lscpu | grep 'L3 cache'")


# In[8]:


get_ipython().system('lscpu | grep MHz')


# In[9]:


get_ipython().system("cat /proc/meminfo | grep 'MemAvailable'")


# In[10]:


get_ipython().system("df -h / | awk '{print $4}'")


# ## Overall Specs:
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Summary as of May 6, 2019**
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CPU: 1xsingle core hyper threaded i.e(1 core, 2 threads) Xeon Processors @2.3Ghz, 46MB Cache
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RAM: ~25.3 GB Available!?
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Disk: ~155 GB Available!
