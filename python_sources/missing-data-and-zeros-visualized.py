#!/usr/bin/env python
# coding: utf-8

# **Goal: for each building and meter pair, visualize where target is missing and where target is zero VS time**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
train['timestamp'] = pd.to_datetime(train.timestamp)
train = train.set_index(['timestamp'])

# Plot missing values per building/meter
f,a=plt.subplots(1,4,figsize=(20,30))
for meter in np.arange(4):
    df = train[train.meter==meter].copy().reset_index()
    df['timestamp'] = pd.to_timedelta(df.timestamp).dt.total_seconds() / 3600
    df['timestamp'] = df.timestamp.astype(int)
    df.timestamp -= df.timestamp.min()
    missmap = np.empty((1449, df.timestamp.max()+1))
    missmap.fill(np.nan)
    for l in df.values:
        if l[2]!=meter:continue
        missmap[int(l[1]), int(l[0])] = 0 if l[3]==0 else 1
    a[meter].set_title(f'meter {meter:d}')
    sns.heatmap(missmap, cmap='Paired', ax=a[meter], cbar=False)


# **Legend:**
# - **X axis:** hours elapsed since Jan 1st 2016, for each of the 4 meter types
# - **Y axis:** building_id
# - **Brown:** meter reading available with non-zero value
# - **Light blue:** meter reading available with zero value
# - **White:** missing meter reading
