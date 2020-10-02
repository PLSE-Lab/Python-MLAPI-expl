#!/usr/bin/env python
# coding: utf-8

# FIFA 19 Analysis - Most promising players acording to overall and evolution

# In[ ]:


import pandas as pd


# In[ ]:


dataset = pd.read_csv('../input/data.csv')


# In[ ]:


dataset.head()


# I've created a new column named **difference** to identify which players has more points to evolute.

# In[ ]:


dataset['difference'] = (dataset['Potential'] - (dataset['Overall']))


# The function **evolution** it's for classify players who is stable (reached the apex), or has a small, medium or big evolution.

# In[ ]:


def evolution(d):
    if d == 0:
        return "Stable"
    elif d >=1 and d<=5:
        return "Small"
    elif d >=6 and d<=10:
        return "Medium"
    elif d >11:
        return "Big"


# In[ ]:


dataset['Evolution'] = dataset['difference'].apply(evolution)


# Now we can filter players with overall > 80 that will have an big evolution (11 or more overall points) in decrescent order.

# In[ ]:


dataset.loc[(dataset['Evolution']== 'Big') & (dataset['Potential']>80)].sort_values(by='Potential', ascending=False)

