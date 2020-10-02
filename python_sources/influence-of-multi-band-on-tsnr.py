#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from json import load
import urllib.request, json 
from pandas.io.json import json_normalize
import seaborn as sns
import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Iterate thour the pages of JSON returned by the API. Note that we are restricting the query to scans with mutliband factor 2 or higher.

# Concatenate everything into one neat DataFrame

# In[ ]:


df = pd.read_csv('../input/mriqc-data-cleaning/bold.csv')


# In[ ]:


df.describe()


# It seems we are working with over 60k scans. Not too shabby!

# In[ ]:


plt.figure(figsize=(10,14))
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='tsnr', data=df, jitter=0.4, alpha=0.3, size=4)


# Lets zoom in on tSNR below 100

# In[ ]:


plt.figure(figsize=(10,14))
sns.stripplot(x='bids_meta.MultibandAccelerationFactor', y='tsnr', data=df[df['tsnr']<100], jitter=0.4, alpha=0.3, size=4)


# **Conclusion: tSNR advantage seems to reverse beyond MB factor 4**
