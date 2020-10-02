#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import commons data libs
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#load data
df = pd.read_csv('../input/ks-projects-201801.csv')
df.columns


# In[ ]:


#check headers and first rows
df.head()


# In[ ]:


#check row
df.loc[df['ID'] == 1000810416]

