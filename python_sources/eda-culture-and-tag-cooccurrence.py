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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from itertools import combinations, chain
from collections import defaultdict, Counter


# In[2]:


df_train = pd.read_csv("../input/train.csv")
df_labels = pd.read_csv("../input/labels.csv")


# In[3]:


df_train.head()


# In[4]:


df_labels.head()


# In[5]:


culture_id_to_name = {row["attribute_id"]: row["attribute_name"] for i, row in df_labels.iterrows() if row["attribute_name"].startswith("culture")}
tag_id_to_name = {row["attribute_id"]: row["attribute_name"] for i, row in df_labels.iterrows() if row["attribute_name"].startswith("tag")}


# In[6]:


def get_train_dict(attribute_ids):
    attribute_ids = [int(i) for i in attribute_ids.split(" ")]
    
    c = [culture_id_to_name[attribute_id] for attribute_id in attribute_ids if attribute_id in culture_id_to_name.keys()]
    t = [tag_id_to_name[attribute_id] for attribute_id in attribute_ids if attribute_id in tag_id_to_name.keys()]
    
    return {"cultures": c, "tags": t}


# In[7]:


train_dict = [get_train_dict(a) for a in df_train.attribute_ids]


# In[8]:


tag_nums = [len(d["tags"]) for d in train_dict]
print(max(tag_nums))
plt.hist(tag_nums, np.arange(0, 10))


# In[9]:


culture_nums = [len(d["cultures"]) for d in train_dict]
print(max(culture_nums))
plt.hist(culture_nums, np.arange(0, 5))


# In[10]:


sorted_culture_pairs = Counter(chain.from_iterable([combinations(d["cultures"], 2) for d in train_dict])).most_common()
sorted_culture_pairs[:20]


# In[11]:


sorted_tag_pairs = Counter(chain.from_iterable([combinations(d["tags"], 2) for d in train_dict])).most_common()
sorted_tag_pairs[:20]


# In[12]:


culture_tag_pair_to_cnt = defaultdict(int)

for d in train_dict:
    for c in d["cultures"]:
        for t in d["tags"]:
            culture_tag_pair_to_cnt[(c, t)] += 1

sorted_culture_tag_pairs = sorted(culture_tag_pair_to_cnt.items(), key=lambda x: x[1], reverse=True)
sorted_culture_tag_pairs[:20]

