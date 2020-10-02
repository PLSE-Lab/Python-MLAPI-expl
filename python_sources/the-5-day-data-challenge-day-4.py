#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# In[12]:


df = pd.read_csv('../input/cereal.csv')


# In[13]:


df.head()


# In[14]:


df_category = df['mfr']


# In[15]:


g = sns.countplot(x=df_category)
g.set_title('Mfr types')


# In[ ]:




