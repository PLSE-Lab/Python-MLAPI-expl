#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/missing-migrants-project/MissingMigrants-Global-2019-12-31_correct.csv')
df.head(4)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isna().sum()

