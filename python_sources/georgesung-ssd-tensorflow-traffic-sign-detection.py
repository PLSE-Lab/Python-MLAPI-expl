#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('git clone https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection.git')


# In[ ]:


import os
os.chdir('ssd_tensorflow_traffic_sign_detection')


# In[ ]:


get_ipython().system('pip install tensorflow==1.13.1')


# In[ ]:


import tensorflow as tf


# In[ ]:


print(tf.__version__)


# In[ ]:


get_ipython().system('pip install moviepy')


# In[ ]:


get_ipython().system('python inference.py -m demo')


# In[ ]:




