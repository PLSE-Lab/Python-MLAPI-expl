#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this competition, we are required to predict the number of `open_channels`, given the electrophysiological signal data (a time series).
# 
# It should be noted that the data are structured in discrete batches of 50 seconds long 10kHz samples (i.e. 500,000 rows per batch), thus we only have 10 batches of samples given in the train set.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

size=16
params = {'legend.fontsize': 'large',
          'figure.figsize': (16,4),
          'axes.labelsize': size*1.1,
          'axes.titlesize': size*1.3,
          'xtick.labelsize': size*0.9,
          'ytick.labelsize': size*0.9,
          'axes.titlepad': 25}
plt.rcParams.update(params)


# In[ ]:


signal_batch_size = 500_000

df_train = pd.read_csv('../input/liverpool-ion-switching/train.csv')

# Add a signal_batch number, for ease of grouping
df_train['signal_batch'] = np.arange(len(df_train)) // signal_batch_size


# ## No. of Labels Per Signal Batch
# 
# `open_channels` have 11 possible values, and each timestamp is associated with a single `open_channels` value.
# 
# From the plot below, we can see that 4 batches only contain 2 different `open_channels` values (i.e. only 0 and 1) whereas 1 of the batch contains up to 11 different `open_channels`. 
# 
# As such, it wouldn't be wise to cross validate based on signal batch, alternative CV set up is required.

# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(12,6))

df_train    .groupby('signal_batch')['open_channels']    .apply(lambda x: len(set(x)))    .value_counts()    .sort_index()    .plot(kind='bar', ax=ax, width=0.8)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', va='bottom', 
                color='black', fontsize=14, 
                #fontweight='heavy',
                xytext=(0,5), 
                textcoords='offset points')

ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels([0,1,2,3,4])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_xlabel('No. of Labels per Signal Batch')
ax.set_ylabel('No. of Signal Batch')
ax.set_title('Distribution of No. of Labels Per Signal Batch '+'$(n = 10)$')

for loc in ['right','top']:
    ax.spines[loc].set_visible(False)


# In[ ]:


def plot_signal_and_label(segment_size=200):
    fig, ax = plt.subplots(1,1, figsize=(14,6))

    sample = np.random.randint(0,9)
    segment = np.random.randint(0,500_000 - segment_size)
    
    df_segment = df_train.query('signal_batch == @sample')
    
    df_segment['signal'].iloc[segment:segment+segment_size]        .plot(ax=ax, label='Signal', alpha=0.8, linewidth=2)
    
    ax_2nd = ax.twinx()
    df_segment['open_channels'].iloc[segment:segment+segment_size]        .plot(ax=ax_2nd, label='Open Channels (Ground Truth)', color='C1', linewidth=2)

    time_start = df_segment['time'].iloc[segment]
    time_end = df_segment['time'].iloc[segment + segment_size-1]
    
    xticklabels = [val for i, val in enumerate(df_segment['time'].iloc[segment:segment + segment_size + 1]) if i%(segment_size//10) == 0]
    xtickloc = [val for i, val in enumerate(df_segment.iloc[segment:segment + segment_size + 1].index) if i%(segment_size//10) == 0]
    
    ax.set_xticks(xtickloc)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Timestamp')
    
    ax.set_ylabel('Signal')
    ax_2nd.set_ylabel('Open Channels')
    
    ax.set_title(f'Signal Batch #{sample} \n('
                 r'$t_{start} = $' + f'${time_start} s, $'
                 r'$t_{end} = $' + f'${time_end} s$' + ')')
    fig.legend(bbox_to_anchor=(1.03,0.5), loc='center left')
    
    ax.spines['top'].set_visible(False)
    ax_2nd.spines['top'].set_visible(False)
    ax.grid(which='major',axis='x', linestyle='--')

    plt.tight_layout()
    plt.show()
    


# ## Signal VS Open Channels
# 
# In the plots below, I sampled segments of 200 continuous timestamps from the randomly selected signal batch, to show the relationship between the `signal` and `open_channels`.
# 
# One observation is that the `open_channels` correlates almost perfectly with the general trend of signal, thus a straight-forward method to approach this problem is using some form of statistical control methods (e.g. moving average based abnomaly detection).

# In[ ]:


for i in range(10):
    plot_signal_and_label(segment_size=200)


# In[ ]:




