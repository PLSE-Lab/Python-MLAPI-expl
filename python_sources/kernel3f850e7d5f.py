#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install bqplot')
get_ipython().system('jupyter nbextension enable --py --sys-prefix bqplot')


# In[ ]:


import numpy as np
from bqplot import Scatter, LinearScale, Figure


# In[ ]:


sc_x, sc_y = LinearScale(), LinearScale()
scatter = Scatter(x=np.random.randn(10), y=np.random.randn(10), scales={'x': sc_x, 'y': sc_y})
Figure(marks=[scatter])

