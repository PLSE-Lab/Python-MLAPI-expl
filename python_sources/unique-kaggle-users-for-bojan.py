#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


users = pd.read_csv("../input/meta-kaggle/Users.csv")
users['Id'].nunique()

