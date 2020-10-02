#!/usr/bin/env python
# coding: utf-8

# In[1]:


#GPU count and name
get_ipython().system('nvidia-smi -L')


# In[2]:


#use this command to see GPU activity while doing Deep Learning tasks, for this command 'nvidia-smi' and for above one to work, go to 'Runtime > change runtime type > Hardware Accelerator > GPU'
get_ipython().system('nvidia-smi')


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


# ## Overall:
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GPU: 1xTesla P100 , having 3584 CUDA cores, 16GB(16.28GB Usable) GDDR6  VRAM
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Tesla P100 Spec Sheet](https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf)
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;CPU: 1xsingle core hyper threaded i.e(1 core, 2 threads) Xeon Processors @2.2Ghz (No Turbo Boost) , 56MB Cache
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RAM: ~15.26 GB Available
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Disk: ~155 GB Available 
