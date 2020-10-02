#!/usr/bin/env python
# coding: utf-8

# Quick notebook showing how to use the [
# Fail-safe, parallel memory reduction](https://www.kaggle.com/wkirgsn/fail-safe-parallel-memory-reduction) utilty script by @wkirgsn.

# In[ ]:


import pandas as pd
import fail_safe_parallel_memory_reduction as reducing


# In[ ]:


# load in the data...
df = pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")

df.head()


# In[ ]:


# ...and make it smaller!
df = reducing.Reducer().reduce(df)


# In[ ]:


# still the same data, though :)
df.head()

