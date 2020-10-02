#!/usr/bin/env python
# coding: utf-8

# ### Given wide format data, get relevant columns
# * Original data was int idy data firmat, here we ahve it in columns

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/500_Cities_CDC.csv")
print(df.shape)
list(df.columns)


# In[ ]:


## Keep only adjusted prevalence columns [i.e "normalized"]
meta_cols = ['StateAbbr', 'PlaceName', 'PlaceFIPS', 'Population2010','Geolocation']
rate_cols = [c for c in df.columns if c.endswith("_AdjPrev")]


# In[ ]:


print(len(rate_cols))
rate_cols


# In[ ]:


KEEP_COLS = meta_cols + rate_cols
df = df[KEEP_COLS]
df.shape


# In[ ]:


# clean latlong
df.Geolocation = df.Geolocation.str.replace("(","").str.replace(")","")


# In[ ]:


df.head()


# In[ ]:


df.to_csv("500_Cities_Health.csv.gz",index=False,compression="gzip")

