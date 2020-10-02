#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[3]:


df = pd.read_csv("../input/microdados_enem_2016_coma.csv", nrows=5, encoding='iso-8859-1')


# In[5]:


for col in df.columns:
    print(col)


# In[ ]:




