#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' cp -a /kaggle/input/catalyst/catalyst/install.sh /tmp/install.sh && chmod 777 /tmp/install.sh && /tmp/install.sh /kaggle/input/catalyst/catalyst')


# In[ ]:


cd /tmp/catalyst


# In[ ]:


get_ipython().system(' pytest')


# In[ ]:


get_ipython().system(' chmod 777 ./bin/check_dl.sh && CUDA_VISIBLE_DEVICES= ./bin/check_dl.sh')


# In[ ]:




