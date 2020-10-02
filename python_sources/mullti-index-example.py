#!/usr/bin/env python
# coding: utf-8

# Multi-Index example

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/gfp2017/GlobalFirePower_multiindex.csv', header=[0,1])


# In[ ]:


df.head()


# In[ ]:


df['Country Data'].head()


# In[ ]:


df['Country Data']['ISO3'].head()


# In[ ]:




