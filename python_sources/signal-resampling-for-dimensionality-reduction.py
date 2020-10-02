#!/usr/bin/env python
# coding: utf-8

# Singal resampling from 800K to 1600 time steps using Scipy signal package and FFT module. I do not have expertise in signal processing. However, this approach could help someone to handle the large sequence data and to do rapid experiments.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_parquet('../input/train.parquet')
print(df_train.shape)


# In[ ]:


train_meta = pd.read_csv('../input/metadata_train.csv')
print(train_meta.shape)
print(train_meta.head(3))


# In[ ]:


train_meta_pos = train_meta[train_meta['target'] == 1]
train_meta_neg = train_meta[train_meta['target'] == 0]
print(train_meta_pos.shape)
print(train_meta_neg.shape)


# In[ ]:


train_meta_pos_p0 = train_meta_pos[train_meta_pos['phase'] == 0]
train_meta_pos_p0.head(10)


# In[ ]:


# Negative signal
train_meta_neg_p0 = train_meta_neg[train_meta_neg['phase'] == 0]
train_meta_neg_p0.head(10)


# Phase 0, Phase 1 and Phase 2 for an id_measurement may or may not have target value 1 i.e. individual phase line will casue discharge. I took Phase 0 signals that has target value 1

# In[ ]:


# Take some samples that has postive and negative target
df_train_sample = df_train.iloc[:, 201:270]
print(df_train_sample.shape)


# In[ ]:


df_train_sample.head(5)


# In[ ]:


# Plot the given data points to see how its trend looks like
df_train_sample.iloc[:, :3].plot()


# In[ ]:


# Function to to FFT and reduce the dimensions
def sample_signals(df):
    cols = df.columns.values
    b, a = signal.butter(4, 0.03, analog=False)
    df_t = []
    for idx in range(df.shape[1]):
        sg = np.squeeze(df.iloc[:, idx:idx+1], axis=1)
        sg_ff = signal.filtfilt(b, a, sg)
        sg_rs = signal.resample(sg_ff, 16*2*50)
        df_t.append(sg_rs)
    df_t = np.asarray(df_t).T
    df_t = pd.DataFrame(df_t, columns=cols)
    return df_t


# In[ ]:


# Show how the processed signals look like
sample1 = sample_signals(df_train_sample.iloc[:,:3])
print(sample1.shape)
sample1.plot()


# In[ ]:


# Comparing given signals and the processed one - Positive signal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
ax1.plot(df_train_sample.iloc[:, :3])
ax2.plot(sample1)


# In[ ]:


# Negative signal - target = 0
sample2 = sample_signals(df_train_sample.iloc[:,3:6])
df_train_sample.iloc[:,3:6].plot()


# In[ ]:


# Comparing given signals and the processed one - Negative signal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
ax1.plot(df_train_sample.iloc[:, 3:6])
ax2.plot(sample2)


# Future work,
# * Check if the preprocessing doesn't lose any information 
# * Identify and emphasize positive and negative signal
# * Any thoughts?
# 
# Please feel free to comment if you have any concerns. Any help in singal processing is appreciated!

# In[ ]:




