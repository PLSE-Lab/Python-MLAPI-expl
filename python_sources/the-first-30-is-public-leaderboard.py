#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


avgs_array = []
for i in range(11):
    avg = train.query(f'open_channels == {i}').signal.mean()
    avgs_array.append(avg)
avgs_array = np.array(avgs_array)
avgs_array


# In[ ]:


def shortest_distance_index(v):
    """Return the index (neg or pos) that has the smallest difference from input v"""
    return np.argmin(np.abs(v - avgs_array))


# In[ ]:


sub.open_channels = test.signal.apply(shortest_distance_index)


# In[ ]:


sub.time = sub.time.apply(lambda x: '{:.4f}'.format(x))
sub.loc[int(sub.shape[0]*0.3):,'open_channels'] = 0
sub.to_csv('submission.csv', index=False)

