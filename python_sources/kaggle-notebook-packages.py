#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('mkdir ./wheatmmdetection')


# In[ ]:


get_ipython().system('cp -r /kaggle/input/wheatmmdetection/* ./wheatmmdetection')


# In[ ]:


get_ipython().run_line_magic('cd', 'wheatmmdetection')


# In[ ]:


get_ipython().system('pip install -r requirements/build.txt')


# In[ ]:


get_ipython().system('pip install /kaggle/input/gwddependencies/pycocotools-2.0-cp37-cp37m-linux_x86_64.whl')


# In[ ]:


get_ipython().system('pip install /kaggle/input/gwddependencies/addict-2.2.1-py3-none-any.whl')


# In[ ]:


get_ipython().system('pip install /kaggle/input/gwddependencies/mmcv-0.6.2-cp37-cp37m-linux_x86_64.whl')

