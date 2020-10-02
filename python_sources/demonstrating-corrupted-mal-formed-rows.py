#!/usr/bin/env python
# coding: utf-8

# # Notebook Purpose
# The purpose of this notebook it to show that there are a few rows in this dataset which are not well-formed. These rows seem to be some sort of summary statistic or addendum referring to previous rows, and they contain only a single column.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


df = pd.read_csv('../input/accepted_2007_to_2018q2.csv/accepted_2007_to_2018Q2.csv')


# In[ ]:


id_column_as_numeric = num_id = pd.to_numeric(df['id'], errors='coerce')


# In[ ]:


# We somehow have IDs which are not integers and get coerced into NaNs!
num_id = pd.to_numeric(df['id'], errors='coerce')
num_id.isna().sum()


# In[ ]:


# what are these non-numerical IDs?
df.loc[num_id.isna(), 'id']


# This is odd, these don't look like IDs at all!

# In[ ]:


# and lo, only the first column is defined, these must be summary statistic rows
# or rows that are follow-ups to the previous row
df.loc[num_id.isna(), :].head()


# In[ ]:


# how should we address this? just drop them!
print(f'df starts out with shape {df.shape}, but actually some rows are mal-formed...')
df = df.loc[num_id.notna(), :]
print(f'we dropped these mal-formed rows, and now we see that the df actually has an effective shape of {df.shape}.')

