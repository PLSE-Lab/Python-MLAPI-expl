#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


import os
os.listdir("/")


# In[ ]:


os.listdir('/kaggle')


# In[ ]:


os.listdir('/kaggle/working')


# In[ ]:


import pandas as pd
pd.DataFrame({1:[2,3],4:[5,6]}).to_csv('whatever.csv')
os.listdir('/kaggle/working')


# In[ ]:


pd.read_csv('/kaggle/working/whatever.csv')


# In[ ]:


os.listdir('/kaggle/config')


# In[ ]:


os.listdir('/kaggle/lib')


# In[ ]:


os.listdir('/kaggle/lib/kaggle')


# In[ ]:


os.listdir('/kaggle/lib/kaggle/competitions')


# In[ ]:


os.listdir('/kaggle/lib/kaggle/competitions/twosigmanews')


# In[ ]:


os.listdir('/home')


# In[ ]:


os.listdir('/mnt')


# In[ ]:


os.listdir('/src')


# In[ ]:


os.listdir('/root')


# In[ ]:


get_ipython().system('cat /root/.profile')


# In[ ]:


get_ipython().system('cat /root/.bashrc')


# In[ ]:


get_ipython().system('ls /root/.keras')


# In[ ]:


get_ipython().system('ls /usr')


# In[ ]:


get_ipython().system('ls /usr/include')


# In[ ]:


get_ipython().system('ls /usr/bin')


# In[ ]:


get_ipython().system('ls /usr/sbin')


# In[ ]:


get_ipython().system('ls -al /root')


# In[ ]:


get_ipython().system('ls -al /root/.keras')


# In[ ]:




