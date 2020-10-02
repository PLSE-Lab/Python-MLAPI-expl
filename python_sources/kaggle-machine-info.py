#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print('# CPU')
get_ipython().system('cat /proc/cpuinfo | egrep -m 1 "^model name"')
get_ipython().system('cat /proc/cpuinfo | egrep -m 1 "^cpu MHz"')
get_ipython().system('cat /proc/cpuinfo | egrep -m 1 "^cpu cores"')

print('\n# OS')
get_ipython().system('cat /etc/*-release')

print('\n# Kernel')
get_ipython().system('uname -a')

print('\n# RAM')
get_ipython().system('cat /proc/meminfo | egrep "^MemTotal"')

print('\n# Nvidia driver')
get_ipython().system('nvidia-smi')

print('\n# Cuda compiler driver')
get_ipython().system('nvcc --version')

