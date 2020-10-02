#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to fill time series gaps in train data set for further analysis, visualization and modelling, and also for convenient visualization of meter values grouped by site/building/meters, with marking of filled values (assuming that zero and missing values contain different information which can help in better predictions).

# In[ ]:


from pathlib import Path

import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.rcParams['figure.figsize'] = (8, 4) # default - (6, 4)
plt.rcParams['figure.max_open_warning'] = 1250 # default - 20


# In[ ]:


DATA_PATH = Path('../input/ashrae-energy-prediction')


# In[ ]:


train_df = pd.read_csv(DATA_PATH/'train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv(DATA_PATH/'test.csv', parse_dates=['timestamp'])
building_df = pd.read_csv(DATA_PATH/'building_metadata.csv')


# One of the differencies between train and test data - there are missing records in train dataset for specific timestamp - building_id - meter values:

# In[ ]:


train_df.groupby('timestamp')['meter'].count().plot()


# We don't have this problem in test dataset:

# In[ ]:


test_df.groupby('timestamp')['meter'].count().plot()


# We can add missing records using the following steps:
# - Set index for timestamp - building_id - meter values
# - Unstack by timestamp, temporarily fill missing values with -1, and stack by timestamp
# - Sort values by timestamp - building_id - meter values like in original train dataset
# - Reset index to sequential values
# - Create boolean indicator (is_metered) whether data is read or not (True/False)
# - Replace filled values with zeros 

# In[ ]:


train_filled_df = (train_df.set_index(['timestamp', 'building_id', 'meter'])
                           .unstack('timestamp')
                           .fillna(-1)
                           .stack('timestamp')
                           .reset_index()
                           .sort_values(['timestamp', 'building_id', 'meter'])
                           .reset_index(drop=True))
train_filled_df['is_metered'] = (train_filled_df['meter_reading'] != -1)
train_filled_df.loc[train_filled_df['meter_reading'] == -1, 'meter_reading'] = 0
train_filled_df


# After this transformation, we have the same distribution of meter counts as in test dataset

# In[ ]:


train_filled_df.groupby('timestamp')['meter'].count().plot()


# Let's add building information for further meter visualization grouped by sites

# In[ ]:


train_filled_full_df = train_filled_df.reset_index().merge(building_df, how='inner', on='building_id')
train_filled_full_df = train_filled_full_df.set_index('index').sort_index()
train_filled_full_df.head()


# We have the following distribution of buildings by sites:

# In[ ]:


train_filled_full_df.groupby(['site_id'])['building_id'].nunique().plot.bar()


# I suggested that it will be convenient to plot meter values grouped by site/building/meter with marking values that we filled before, to separate zero and missing values.  
# Also may be it will be useful to see the patterns of meter reading.
# The code for plotting of multicolored lines is used from this blog - http://abhay.harpale.net/blog/python/how-to-plot-multicolored-lines-in-matplotlib/, with a few modifications, to plot meter values as subplots.

# In[ ]:


def find_contiguous_colors(colors):
    segs = []
    curr_seg = []
    prev_color = ''
    for c in colors:
        if c == prev_color or prev_color == '':
            curr_seg.append(c)
        else:
            segs.append(curr_seg)
            curr_seg = []
            curr_seg.append(c)
        prev_color = c
    segs.append(curr_seg)
    return segs
 
def plot_multicolored_lines(x,y,colors,ax=None,**kwargs):
    segments = find_contiguous_colors(colors);
    start= 0
    
    for seg in segments:
        end = start + len(seg)
        if ax is None:
            l, = plt.gca().plot(x[start:end],y[start:end],lw=2,c=seg[0], **kwargs)
        else:
            l, = ax.plot(x[start:end],y[start:end],lw=2,c=seg[0], **kwargs)
        start = end


# Now you can specify site_id and visualize all meter values grouped by building/meter:

# In[ ]:


site_id = 10

site_df = train_filled_full_df[train_filled_full_df['site_id']==site_id]
buildings = site_df['building_id'].unique().tolist()

for building_id in buildings:
    building_meter_df = site_df[site_df['building_id']==building_id].set_index('timestamp')
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 4))
    fig.suptitle(f'building_id={building_id}')
    
    for key, group in building_meter_df.groupby('meter'): 
        ax = axes[int(key)]
        ax.set_title(f'meter={key}')
        
        if len(group) > 0:
            colors = group['is_metered'].map({True: 'green', False: 'red'})
            plot_multicolored_lines(group.index, group['meter_reading'], colors, ax=ax)


# I hope that it will be useful :)
