#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#GPU count and name
get_ipython().system('nvidia-smi -L')


# In[ ]:


#use this command to see GPU activity while doing Deep Learning tasks, for this command 'nvidia-smi' and for above one to work, go to 'Runtime > change runtime type > Hardware Accelerator > GPU'
get_ipython().system('nvidia-smi')


# In[ ]:


get_ipython().system("lscpu |grep 'Model name'")


# In[ ]:


#no.of sockets i.e available slots for physical processors
get_ipython().system("lscpu | grep 'Socket(s):'")


# In[ ]:


#no.of cores each processor is having 
get_ipython().system("lscpu | grep 'Core(s) per socket:'")


# In[ ]:


#no.of threads each core is having
get_ipython().system("lscpu | grep 'Thread(s) per core'")


# In[ ]:


get_ipython().system('lscpu | grep "L3 cache"')


# In[ ]:


#if it had turbo boost it would've shown Min and Max MHz also but it is only showing current frequency this means it always operates at 2.3GHz
get_ipython().system('lscpu | grep "MHz"')


# In[ ]:


#memory that we can use
get_ipython().system("cat /proc/meminfo | grep 'MemAvailable'")


# In[ ]:


#hard disk that we can use
get_ipython().system("df -h / | awk '{print $4}'")

