#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pandas --upgrade')
import importlib
importlib.invalidate_caches()


# In[ ]:


import pandas as pd
get_ipython().system('pip show pandas')
print(pd.__version__) # NOT WORKING

