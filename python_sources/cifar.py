#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' cp -a /kaggle/input/catalyst/catalyst/install.sh /tmp/install.sh && chmod 777 /tmp/install.sh && /tmp/install.sh /kaggle/input/catalyst/catalyst')


# In[ ]:


cd /tmp/catalyst/examples


# In[ ]:


get_ipython().system(' catalyst-dl run --config cifar_simple/config.yml')


# In[ ]:




