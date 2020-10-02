#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[7]:


import json

file = pd.read_json("../input/grocery items bounding boxes.json", lines=True)


# In[11]:


file.to_csv("output.csv")


# In[ ]:




