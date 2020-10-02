#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import collections


# In[ ]:


data = pd.read_csv("../input/train.csv")
b = data


# In[ ]:


def get_day(row):
    d_d = collections.defaultdict(lambda : 0)
    for value in row:
        val = int(value)
        val_w = (val - 1) % 7 + 1
        d_d[val_w] += ((val - 1) // 7 + 1)
    return sorted(list(d_d.items()), key = lambda x: x[1], reverse = True)[0][0]
    
a = b.visits.str.split().apply(get_day)


# In[ ]:


c = pd.concat([b.id, a], keys = ["id", "nextvisit"], axis = 1)
c.to_csv("solution.csv", index = False)

