#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install albumentations > /dev/null')


# In[ ]:


import albumentations


# In[ ]:


albumentations.__version__


# In[ ]:


from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()
[d.name for d in local_device_protos if d.device_type == 'GPU']


# In[ ]:




