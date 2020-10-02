#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


df = pd.read_csv('/kaggle/input/cusersmarildownloadsearningcsv/earning.csv',delimiter=';',index_col=[0])


# In[ ]:


df['ratioprofessionals'] = df['femaleprofessionals']/df['maleprofessionals']
df[['femaleprofessionals','maleprofessionals']]


# In[ ]:




