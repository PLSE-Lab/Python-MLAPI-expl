#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

import os
print(os.listdir("../input"))


# In[ ]:


data=pd.read_json("../input/recipes.json",lines=True)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data['Author'].value_counts()[0:10]


# In[ ]:




