#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_json("../input/roam_prescription_based_prediction.jsonl", lines=True)


# In[11]:


data.head(30)


# In[4]:


provider_data = pd.DataFrame([v for v in data["provider_variables"]])


# In[5]:


provider_data.head()


# In[6]:


provider_data.groupby("specialty")["years_practicing"].mean().sort_values(ascending=False)


# In[7]:


from collections import Counter

rx_counts = Counter()

for rx_dist in data.cms_prescription_counts:
    rx_counts.update(rx_dist)

rx_series = pd.Series(rx_counts)

rx_series.sort_values(ascending=False)


# In[8]:


def merge_counts(dicts):
    merged = Counter()
    
    for d in dicts:
        merged.update(d)
    return merged.most_common(20)

merged_data = pd.concat([data, provider_data], axis=1)

merged_data.groupby("specialty")["cms_prescription_counts"].apply(merge_counts)


# In[9]:


provider_data["specialty"].value_counts()


# In[10]:


merged_data.groupby("specialty")["cms_prescription_counts"].apply(merge_counts)["General Practice"]


# In[ ]:




