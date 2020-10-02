#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


test_df = pd.read_csv("../input/train.csv",nrows=5)
test_df.head()


# In[9]:


test_df.dtypes


# In[10]:


df = pd.read_csv("../input/train.csv")
df.head()


# In[13]:


df.describe().round(2)


# In[27]:


df['Age'].plot.hist()


# In[ ]:




