#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Install offline tensorflow == 1.14.0 
# make sure internet is off

# ### step1 :copy tf-1140 to tmp/pip/cache/...
# ### step2 :install tensorflow==1.14.0

# In[ ]:


get_ipython().system('mkdir -p /tmp/pip/cache/')
get_ipython().system('cp -r ../input/tf-1140/ /tmp/pip/cache/tensorflow14')
get_ipython().system('pip install --no-index --find-links /tmp/pip/cache/tensorflow14 tensorflow==1.14.0')


# In[ ]:


# Check tensorflow version
import tensorflow as tf
print('Current tensorflow version is {}'.format(tf.__version__))

