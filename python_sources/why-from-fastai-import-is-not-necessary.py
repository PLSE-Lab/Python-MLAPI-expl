#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# actually only import version module
from fastai import *


# In[ ]:


# try untar and tab, you get nothing
dir()


# In[ ]:


version


# In[ ]:


# there are only two attributes in version module
get_ipython().run_line_magic('pinfo2', 'version')


# In[ ]:


version.__all__


# In[ ]:


version.__version__


# In[ ]:


# actually only get __version__ from version module
import fastai


# In[ ]:


dir()


# In[ ]:



get_ipython().run_line_magic('pinfo2', 'fastai')


# In[ ]:


# type fastai. and tab, only get fastai.version
fastai.version


# # Restart session

# In[1]:


from fastai.vision import *


# In[2]:


dir()


# In[3]:


__version__


# So, `from fastai import *` does not add anything additional to `from fastai.vision import *`. Therefore, it is unnecessary to do `from fastai import *`.

# In[ ]:




