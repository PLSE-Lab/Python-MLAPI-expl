#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
PATH="../input/iwildcam-2020-fgvc7"
os.listdir(PATH)


# In[ ]:


import pandas as pd
import numpy as np
from fastai.vision import *
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)


# In[ ]:


get_ipython().system(" ls '../input/iwildcam-2020-fgvc7/train' | wc")


# In[ ]:


get_ipython().system(" du -sh '../input/iwildcam-2020-fgvc7/train'")


# In[ ]:


get_ipython().system(" ls '../input/iwildcam-2020-fgvc7/test' | wc")


# In[ ]:


get_ipython().system(" du -sh '../input/iwildcam-2020-fgvc7/test'")


# In[ ]:


import numpy as np
from fastai.vision import *
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)


# In[ ]:


path = Path('../input/iwildcam-2020-fgvc7/train')


# to be continue...
