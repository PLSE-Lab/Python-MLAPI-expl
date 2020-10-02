#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch


# In[ ]:


torch.cuda.current_device()


# In[ ]:


torch.cuda.device(0)


# In[ ]:


torch.cuda.device_count()


# In[ ]:


torch.cuda.get_device_name(0)


# In[ ]:


torch.cuda.is_available()


# In[ ]:


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:




