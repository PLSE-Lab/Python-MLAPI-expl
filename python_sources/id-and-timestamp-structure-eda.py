#!/usr/bin/env python
# coding: utf-8

# Let's look at the timestamp...

# In[ ]:


import pandas as pd
import numpy as np # linear algebra
from matplotlib import pyplot as plt
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")
ids = df['id'].values
timestamps = df['timestamp'].values


# In[ ]:


print('Ids count:', len(np.unique(ids)), 'Min id:', ids.min(), 'Max id:', ids.max())
print('Time frames (unique timestamps) count:', len(np.unique(timestamps)), 'Min:', timestamps.min(), 'Max:', timestamps.max())


# Id values span from 0 to 2158, but we only have 1424 unique ids. Let's visualise the missing. 

# In[ ]:


full_range = range(ids.max()+1)
unique_ids, unique_id_counts = np.unique(ids, return_counts=True)
missing_values = [x for x in full_range if not x in unique_ids]
print('Missing count', len(missing_values))
plt.figure(figsize=(9,3))
plt.ylabel('Timestamp')
plt.plot(unique_ids, '.b',         missing_values, '.r')
plt.show()


# So in the range of 0 to 2158, there are 735 ids that are missing. These are equally distributed across the whole range (from 0 to 2158). If these ids are in the test set, this might be very important important. If someone already knows if that is actually the case, please comment.

# In[ ]:


id_count = [len(df[df['timestamp'] == i]['id'].unique()) for i in range(timestamps.max()+1)]
plt.figure(figsize=(9,3))
plt.xlabel('Timestamp index')
plt.ylabel('Ids count')
plt.plot(range(timestamps.max()+1), id_count,'.b')
plt.show()


# Indeed, the figure shows that for each timestamp there is a variable amount of ids. This means that some ids are living longer than others. 

# In[ ]:


print('Min', unique_id_counts.min(), 'Max', unique_id_counts.max(),       'Mean',unique_id_counts.mean(),  'Median', np.median(unique_id_counts))
plt.figure(figsize=(9,6))
plt.plot(unique_ids, unique_id_counts, '.r')
print('Ids with max timestamps alive', len(np.where(unique_id_counts == unique_id_counts.max())[0]))
plt.xlabel('Ids')
plt.ylabel('Timestamps alive')
plt.show()


# There are 527 ids with the maximum life span of 1813 timestamps. Furthermore, one can clearly see the pattern. There are groups of ids that have the same life span. Let's look at this groups in more detail.

# In[ ]:


id_groups = {}
for _id in unique_ids:
    key = tuple(sorted(df[df['id'] == _id]['timestamp'].values))
    if not key in id_groups:
        id_groups[key] = []
    id_groups[key].append(_id)


# In[ ]:


print('Groups count', len(id_groups.keys()))
ids_in_group = []
timestamps_in_group = []
for key in id_groups.keys():
    ids_in_group.append(len(id_groups[key]))
    timestamps_in_group.append(len(key))
ids_in_group = np.array(ids_in_group)
timestamps_in_group = np.array(timestamps_in_group)
index = np.argsort(ids_in_group)[::-1]

for i in range(30):
    print('Group', i+1, 'has',ids_in_group[index][i],'ids and',timestamps_in_group[index][i],'timestamps.')


# In[ ]:





# In[ ]:




