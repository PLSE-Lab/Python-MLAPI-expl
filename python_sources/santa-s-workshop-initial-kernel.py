#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import numpy as np
from fastai.tabular import *
import matplotlib.pyplot as plt
import seaborn as sns


# # Define paths

# In[ ]:


path = Path('../input/santa-workshop-tour-2019')


# In[ ]:


path.ls()


# # Load data

# In[ ]:


df = pd.read_csv(path/'family_data.csv')
samplesub = pd.read_csv(path/'sample_submission.csv')


# In[ ]:


df.head()


# # Exploratory data analysis

# ### Let's check how big are the families in our dataset

# In[ ]:


grouped_families = df.groupby('n_people')['family_id'].count().reset_index()


# In[ ]:


grouped_families.head()


# In[ ]:


sns.barplot(x='n_people',y='family_id',data=grouped_families)


# We have a good range of families sizes, from just 2 (couples) to very large families with 8 members. However, the majority of families have between 3 and 5 members.

# In[ ]:




