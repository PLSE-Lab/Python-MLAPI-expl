#!/usr/bin/env python
# coding: utf-8

# Old Newspapers
# ====

# In[ ]:


import pandas as pd


# Read the corpus into a `pandas` DataFrame.
# 
# It'll take a while to load them into the kernel, ~15 mins =)

# In[ ]:


df = pd.read_csv('../input/old-newspaper.tsv', sep='\t', error_bad_lines=False)


# Here's a sample of the DataFrame object:

# In[ ]:


df.head(10)


# The list of languages available:

# In[ ]:


languages = df['Language'].unique()
languages


# In[ ]:




