#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


get_ipython().system('pip install fastparquet')


# In[ ]:


stackoverflow_posts = pd.read_parquet('../input/get-stackexchange-archival-data/stackoverflow_posts.parquet.gzip', engine='fastparquet')


# In[ ]:


stackoverflow_posts.to_csv('stackoverflow_posts.csv', index=False)

