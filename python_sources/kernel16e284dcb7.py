#!/usr/bin/env python
# coding: utf-8

# # Most Common Random Seeds
# 
# A long time ago (a year and a half ago, to be exact), aroundw ehn the GitHub code repository first went up in the Google BigQuery sample datasets, I did a little project with that data trying to determine what the most common random seeds in the Python community are.
# 
# [I wrote the result up on my personal blog afterwards](http://www.residentmar.io/2016/07/08/randomly-popular.html). Basically, I had a hunch that 42 would represent itself very well in the outcome (because [programmers are weird](https://en.wikipedia.org/wiki/42_(number)). Now that this dataset is available on Kaggle, let's see if we can replicate my result!

# In[6]:


from google.cloud import bigquery
client = bigquery.Client()

QUERY = (
"""SELECT REGEXP_EXTRACT(content, r'(random_state=\d*|seed=\d*|random_seed=\d*|random_number=\d*)') FROM `bigquery-public-data.github_repos.sample_contents`"""
)

query_job = client.query(QUERY)

iterator = query_job.result(timeout=30)
rows = list(iterator)


# This nets us close to 3 million random seeds.

# In[7]:


len(rows)


# In[12]:


import pandas as pd
seeds = pd.Series(rows)


# In[29]:


import numpy as np
seeds = (
    seeds.map(lambda s: s[0])
     .map(lambda s: np.nan if pd.isnull(s) else s.split("=")[-1].strip())
     .map(lambda s: np.nan if (pd.isnull(s) or s == "") else float(s))
     .value_counts(dropna=False)
)


# In[55]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

(seeds
     # Reagg by percentage usage, excluding the empty (NaN) seed.
     .pipe(lambda srs: srs / srs.iloc[1:].sum())
     .sort_values(ascending=False)
     # Exclude the empty (NaN) seed.
     .head(11)
     .iloc[1:]
     .pipe(lambda srs: srs.reindex(srs.index.astype(int)))
     .plot.bar(
         figsize=(12, 6), title='Most Common Integer Random Seeds'
     )
)
ax = plt.gca()
ax.set_ylabel('Percent Usage')
ax.set_xlabel('Seed (Integer)')
pass


# It looks like 42 is indeed an extremely popular random seed (~4% of all random seeds used), coming in third behind the obvious 0 and 1, and followed distantly by [variations on the combination on my luggage](https://www.youtube.com/watch?v=a6iW-8xPw3k).
