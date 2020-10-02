#!/usr/bin/env python
# coding: utf-8

# Step 1: Load `time_to_failure` from training dataset

# In[1]:


import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('../input/train.csv', dtype={'time_to_failure': np.float32}, usecols=[1])
train.head()


# Step 2: Define bin (chunk) duration and determine sample (frame) duration from it. Define more precise intervals between earthquakes.

# In[13]:


T_chunk = 0.001064
S_block = 1280
S_chunk = 4096
frames_per_chunk = S_chunk - 1/S_block
T_frame = T_chunk/frames_per_chunk
ttf_diffs = [1.4697, 11.54102, 14.18108, 8.85708, 12.69404, 8.056066, 7.05905, 16.10807, 7.906067, 9.63804, 11.42708, 11.02503, 8.829078, 8.56705, 14.75209345, 9.4601, 11.619]
ttf_lower_bound = 0.000556

epsilon = 0.00063


# Step 3: Validate that we can asssume equal time between samples (`T_frame`) and times calculated with this assumption are always within 0.63 milliseconds from `time_to_faulure` values in training dataset. (Print failing samples otherwise, empty log is expected, exept for maybe earthquake moments)

# In[14]:


#size = 150000
size = 4096
segments = (train.shape[0]) // size

#Y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

ttf_idx = 0
y = 0
start = 0
for segment in tqdm(range(segments)):
    seg = train.iloc[start:start+size]
    y -= size * T_frame
    while y < ttf_lower_bound:
        y += ttf_diffs[ttf_idx]
        ttf_idx += 1
    start += size

    diff = y - seg['time_to_failure'].values[-1]
    if abs(diff) > epsilon:
        print(ttf_idx, segment, diff, y, seg['time_to_failure'].values)
#    Y_tr.loc[segment, 'time_to_failure'] = y

#Y_tr.to_csv('time_prediction.csv', index=False)


# Conclusion:
# - time in training dataset can be considered contiguous with 0.63 milliseconds precision.
# - time between bins is 1.064 milliseconds. 1ms and 1.1ms differences seen in source data are not randomly spread, but distributed corresponding to rounding to 0.1 precision.
