#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

all_primary_metadata = pd.read_csv('/kaggle/input/covid19-soles/all_primary_metadata.csv',header=0)
all_primary_refs = pd.read_csv('/kaggle/input/covid19-soles/all_primary_refs.csv',header=0)
all_primary_data = pd.read_csv('/kaggle/input/covid19-soles/all_primary_data.csv',header=0)
all_non_primary_metadata = pd.read_csv('/kaggle/input/covid19-soles/all_non_primary_metadata.csv',header=0)


# In[ ]:


all_primary_data = all_primary_data[['Title','Journal','objective', 'method', 'detail_method', 'subjects','peer_reviewed','DOI']]
all_primary_data.head(50)


# In[ ]:


all_primary_data = pd.read_csv('/kaggle/input/covid19-soles/all_primary_data.csv',header=0)
all_primary_data.columns.values


# In[ ]:


all_primary_refs.columns.values


# In[ ]:


all_primary_metadata.columns.values


# In[ ]:


all_non_primary_metadata.columns.values


# In[ ]:




