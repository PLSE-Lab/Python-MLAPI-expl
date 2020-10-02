#!/usr/bin/env python
# coding: utf-8

# This kernel attempts to plot the likelihood that a combination of band 1 and band 2 at a certain angle is an iceberg or ship.
# 
# This did not prove particularly useful - I attemted to use the likelihood that each pixel would be an iceberg as a third channel and did not see improvement.
# 
# I am just publishing this in the hope it helps someone.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt


# In[ ]:


train = pd.read_json("../input/train.json")


# Remove records missing _inc angle_

# In[ ]:


train_clean = train[train['inc_angle'] != 'na']
train_clean['inc_angle'] = train_clean['inc_angle'].astype(np.float32)


# Flatten to a 2D array where each row contains _band 1_, _band 2_, _inc angle_ and _is iceberg_ for processing

# In[ ]:


def get_band_coords(band_1, band_2, inc_angle, is_iceberg):
    band_1 = np.array(band_1)
    band_2 = np.array(band_2)
    inc_angle = np.array([inc_angle])
    is_iceberg = np.array([is_iceberg])
    return np.stack(np.broadcast(band_1, band_2, inc_angle, is_iceberg))

band_labels = np.concatenate([
    get_band_coords(row['band_1'], row['band_2'], row['inc_angle'], row['is_iceberg'])
    for _, row in train_clean.iterrows()
])

band_1 = band_labels[:, 0]
band_2 = band_labels[:, 1]
inc_angles = band_labels[:, 2]
is_iceberg = band_labels[:, 3]


# Clip bands to restrict the size of the final grids

# In[ ]:


band_1_clipped = band_1.clip(band_1.mean() - band_1.std() * 2, band_1.mean() + band_1.std())
band_2_clipped = band_2.clip(band_2.mean() - band_2.std() * 2, band_2.mean() + band_2.std())


# Round bands to 1 decimal to create buckets and convert band values to integer coordinates for averaging and rendering

# In[ ]:


band_1_rounded = np.round(band_1_clipped, 1)
band_2_rounded = np.round(band_2_clipped, 1)
band_1_buckets = np.unique(band_1_rounded)
band_2_buckets = np.unique(band_2_rounded)
band_1_mapped = (band_1_rounded - band_1_buckets.min()) * 10
band_1_mapped = band_1_mapped.astype(np.int)
band_2_mapped = (band_2_rounded - band_2_buckets.min()) * 10
band_2_mapped = band_2_mapped.astype(np.int)


# Map inc angles to buckets (I manually created the buckets)

# In[ ]:


INC_ANGLE_BUCKET_COUNT = 10

def map_inc_angle(inc_angle):
    if inc_angle < 35: return 0
    elif inc_angle < 36: return 1
    elif inc_angle < 37: return 2
    elif inc_angle < 38: return 3
    elif inc_angle < 39: return 4
    elif inc_angle < 40: return 5
    elif inc_angle < 41: return 6
    elif inc_angle < 42: return 7
    elif inc_angle < 43: return 8
    else: return 9
    
inc_angles_mapped = np.array(
    [ map_inc_angle(inc_angle) for inc_angle in inc_angles]
)


# Count number of times that the combination of any values of _band 1_ and _band 2_ at a given _inc angle_ are a ship or iceberg

# In[ ]:


mapped_icebergs = np.stack([band_1_mapped, band_2_mapped, inc_angles_mapped, is_iceberg.astype(np.int)], axis=1)
bucket_counter = np.zeros((len(band_1_buckets), len(band_2_buckets), INC_ANGLE_BUCKET_COUNT, 2))
for i in range(mapped_icebergs.shape[0]):
    x, y, ang, ib = mapped_icebergs[i, :]
    bucket_counter[x, y, ang, ib] += 1


# Get the ratio of iceberg to total for each (_band 1_, _band 2_, _inc angle_) tuple and set any empty or rare entries to *0.5*

# In[ ]:


ship_counters = bucket_counter[:, :, :, 0]
iceberg_counters = bucket_counter[:, :, :, 1]
total_counts = ship_counters + iceberg_counters
material_grid = iceberg_counters / (ship_counters + iceberg_counters)
material_grid[np.isnan(material_grid)] = 0.5
material_grid[total_counts < 10] = 0.5


# Graph the results - yellow indicates this combination of bands at this angle is likely an iceberg.
# 
# Yellow = Likely iceberg
# Blue = Likely ship
# X axis = band 1
# Y axis = band 2
# Inc Angle - increases left to right, top to bottom

# In[ ]:


plt.figure(figsize=(15,15))
for x in range(INC_ANGLE_BUCKET_COUNT):
    plt.subplot(INC_ANGLE_BUCKET_COUNT / 3 + 1, 3, x+1)
    materials = material_grid[:, :, x]
    plt.imshow(materials)


# Let's add some blur to make them look nice

# In[ ]:


from scipy.ndimage.filters import gaussian_filter
plt.figure(figsize=(15,15))
for x in range(INC_ANGLE_BUCKET_COUNT):
    plt.subplot(INC_ANGLE_BUCKET_COUNT / 3 + 1, 3, x+1)
    materials = material_grid[:, :, x]
    materials = gaussian_filter(materials, 3, mode='nearest')
    plt.imshow(materials)


# Conclusion
# ----
# 
# The incidence angle is important and the graphs look neat. That's about it...

# In[ ]:




