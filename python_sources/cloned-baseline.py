#!/usr/bin/env python
# coding: utf-8

# Simple average of two kernels

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import math


# # take 1
# 
# Based on https://www.kaggle.com/suicaokhoailang/an-embarrassingly-simple-baseline-0-960-lb

# In[ ]:


sub1 = pd.read_csv("../input/liverpool-ion-switching/test.csv")

n_groups = 40
sub1["group"] = 0
for i in range(n_groups):
    ids = np.arange(i*50000, (i+1)*50000)
    sub1.loc[ids,"group"] = i


# In[ ]:


for i in range(n_groups):
    sub = sub1[sub1.group == i]
    signals = sub.signal.values
    imax, imin = math.floor(np.max(signals)), math.ceil(np.min(signals))
    signals = (signals - np.min(signals))/(np.max(signals) - np.min(signals) + 0.075)
    signals = signals*(imax-imin)
    sub1.loc[sub.index,"open_channels"] = np.array(signals,np.int)


# # take 2
# Based on https://www.kaggle.com/konradb/liverpool-averaging-median

# In[ ]:


def shortest_distance_index(v):
    """Return the index (neg or pos) that has the smallest difference from input v"""
    return np.argmin(np.abs(v - avgs_array))


# In[ ]:


train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
sub2 = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')

avgs_array = []

for i in range(11):
    avg = train.query(f'open_channels == {i}').signal.median()
    avgs_array.append(avg)

avgs_array = np.array(avgs_array)

sub2.open_channels = test.signal.apply(shortest_distance_index)


# In[ ]:


idx = np.where(sub1['open_channels'] != sub2['open_channels'])[0]


# In[ ]:


# combine
sample_df = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv", dtype={'time':str})
sample_df['o1'] = np.array(sub1.open_channels)
sample_df['o2'] = np.array(sub2.open_channels)

sample_df['open_channels'] = (0.7 * sample_df['o1'] + 0.3 * sample_df['o2']).astype(int)


# In[ ]:


sample_df['diff'] = sample_df['open_channels'] - sample_df['o1']
sample_df['diff'].value_counts()


# In[ ]:


sample_df.head(3)


# In[ ]:


sample_df[['time', 'open_channels']].to_csv("submission.csv",index=False)

