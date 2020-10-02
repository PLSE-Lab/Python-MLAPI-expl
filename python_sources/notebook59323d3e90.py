#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jupyter nbextension enable --py --sys-prefix widgetsnbextension')
import pandas as pd
import numpy as np
import ipywidgets


# In[ ]:


def f(x):
    return x


# In[ ]:


ipywidgets.interact(f, x=10);


# In[ ]:


ipywidgets.__version__


# In[ ]:




