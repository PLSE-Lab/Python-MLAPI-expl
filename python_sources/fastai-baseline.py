#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# load data
train_df = pd.read_csv('../input/train_relationships.csv')
test_df = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()

