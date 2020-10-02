#!/usr/bin/env python
# coding: utf-8

# # Handling noise in batch7
# I was bothered by batch 7.
# I'm sure everyone does, too.
# 
# As we all know, there is a strong spike in batch 7 that prevents the model from learning.  
# (https://www.kaggle.com/c/liverpool-ion-switching/discussion/149846)
# 
# When I analyzed batch 7 in detail, I found it to be due to an artificial error.
# 
# So, I took the method of replacing the outliers with appropriate values.
# 
# 

# # Const

# In[ ]:


PATH_TRAIN = '/kaggle/input/data-without-drift/train_clean.csv'


# # Import everything I need :)

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # My function

# In[ ]:


def group_feat_train(_train):
    train = _train.copy()
    # group init
    train['group'] = int(0)

    # group 1
    idxs = (train['batch'] == 3) | (train['batch'] == 7)
    train['group'][idxs] = int(1)

    # group 2
    idxs = (train['batch'] == 5) | (train['batch'] == 8)
    train['group'][idxs] = int(2)

    # group 3
    idxs = (train['batch'] == 2) | (train['batch'] == 6)
    train['group'][idxs] = int(3)

    # group 4
    idxs = (train['batch'] == 4) | (train['batch'] == 9)
    train['group'][idxs] = int(4)
    
    return train[['group']]


# # Preparation

# setting

# In[ ]:


sns.set()


# <br>
# 
# load dataset

# In[ ]:


df_tr = pd.read_csv(PATH_TRAIN)


# <br>
# 
# add batch and group

# In[ ]:


batch_list = []
for n in range(10):
    batchs = np.ones(500000)*n
    batch_list.append(batchs.astype(int))
batch_list = np.hstack(batch_list)
df_tr['batch'] = batch_list


# In[ ]:


group = group_feat_train(df_tr)
df_tr = pd.concat([df_tr, group], axis=1)


# # EDA
# Let's plot the batch 7 signals by open_channels.

# In[ ]:


df_tr['mean_sig'] = df_tr.groupby(['open_channels','batch'])['signal'].transform('mean')


# In[ ]:


res = 100
x = np.arange(len(df_tr))

fig = plt.figure(figsize=(20, 5))
for i in df_tr[df_tr['batch']==7]['open_channels'].unique():
    idxs = (df_tr['batch'] == 7) & (df_tr['open_channels'].values==i)
    plt.scatter(x[idxs], df_tr['signal'].values[idxs], s=2, label=f'open_channel: {i}')
    
fig.legend(fontsize=10)
plt.title('batch7')
plt.ylabel('signal')


# ---> This result was not what I had imagined.
# 
# <br>
# Let's look at it in further detail.

# In[ ]:


res = 1
x = np.arange(len(df_tr))

fig = plt.figure(figsize=(20, 5))
for i in df_tr[df_tr['batch']==7]['open_channels'].unique():
    idxs = (df_tr['batch'] == 7) & (df_tr['open_channels'].values==i)
    plt.scatter(x[idxs], df_tr['signal'].values[idxs], s=4, label=f'open_channel: {i}')
    
fig.legend(fontsize=10)
plt.xlim(3_642_000, 3_650_000)
plt.title('batch7')
plt.ylabel('signal')


# ---> There is too much noise.  
# ---> The signal for open_channels=3 has been present at low values below -2.    
# ---> I decided to replace the outliers that exist in index=3641000 ~ 3829000 with other appropriate values.  

# <br>
# 
# For example, replace the value of open_channel=3(blue points).    
# The replacement value is the average value of open_channel=3, except for index=3641000~3829000.  

# In[ ]:


left = 3641000
right = 3829000


# Mean value visualization

# In[ ]:


res = 100
x = np.arange(len(df_tr))

fig = plt.figure(figsize=(20, 5))
for i in df_tr[df_tr['batch']==7]['open_channels'].unique():
    idxs = (df_tr['batch'] == 7) & (df_tr['open_channels'].values==i)
    plt.scatter(x[idxs], df_tr['signal'].values[idxs], s=2, label=f'open_channel: {i}')
plt.axvline(left, linestyle='--', color='black')
plt.axvline(right, linestyle='--', color='black')

# batch7 signal mean (without noisy area)
idxs_noisy = (df_tr['open_channels']==3) & (left<df_tr.index) & (df_tr.index<right)
idxs_not_noisy = (df_tr['open_channels']==3) & ~idxs_noisy
plt.axhline(df_tr[idxs_not_noisy]['signal'].mean(), label='mean(open_channel=3)', color='black')
    
fig.legend(fontsize=10)
plt.title('batch7')
plt.ylabel('signal')


# <br>
# Determine the threshold for outliers.  
# In this case, the threshold was determined by eye measurement.

# In[ ]:


thresh_high = 2.0
thresh_low = 0.1


# In[ ]:


res = 1
x = np.arange(len(df_tr))

fig = plt.figure(figsize=(20, 5))
for i in df_tr[df_tr['batch']==7]['open_channels'].unique():
    idxs = (df_tr['batch'] == 7) & (df_tr['open_channels'].values==i)
    plt.scatter(x[idxs], df_tr['signal'].values[idxs], s=4, label=f'open_channel: {i}')
    
plt.axvline(left, linestyle='--', color='black')

# batch7 signal mean (without noisy area)
idxs_noisy = (df_tr['open_channels']==3) & (left<df_tr.index) & (df_tr.index<right)
idxs_not_noisy = (df_tr['open_channels']==3) & ~idxs_noisy
mean = df_tr[idxs_not_noisy]['signal'].mean()
plt.axhspan(thresh_high, thresh_low, color='gray', alpha=0.3)
plt.axhline(mean, label='mean(open_channel=3)', color='black')
    
fig.legend(fontsize=10)
plt.xlim(3_642_000, 3_650_000)
plt.title('batch7')
plt.ylabel('signal')


# ---> The signal of open_channels=3, which is out of the gray shading, is taken as an outlier.
# 
# <br>
# Let's replace the outlier with the average value.

# In[ ]:


df_tr['signal_mod'] = df_tr['signal'].values

idxs_outlier = idxs_noisy & (thresh_high<df_tr['signal'].values)
df_tr['signal_mod'][idxs_outlier]  = mean
idxs_outlier = idxs_noisy & (df_tr['signal'].values<thresh_low)
df_tr['signal_mod'][idxs_outlier]  = mean


# In[ ]:


res = 1
x = np.arange(len(df_tr))

fig = plt.figure(figsize=(20, 5))
for i in df_tr[df_tr['batch']==7]['open_channels'].unique():
    idxs = (df_tr['batch'] == 7) & (df_tr['open_channels'].values==i)
    plt.scatter(x[idxs], df_tr['signal_mod'].values[idxs], s=4, label=f'open_channel: {i}')
    
plt.axvline(left, linestyle='--', color='black')

# batch7 signal mean (without noisy area)
idxs_noisy = (df_tr['open_channels']==3) & (left<df_tr.index) & (df_tr.index<right)
mean = df_tr[idxs_noisy]['signal'].mean()
plt.axhspan(thresh_high, thresh_low, color='gray', alpha=0.3)
# plt.axhline(mean, label='mean(open_channel=3)', color='black')
    
fig.legend(fontsize=10)
plt.xlim(3_642_000, 3_650_000)
# plt.xlim(3_642_000, 3_900_000)
plt.title('batch7')
plt.ylabel('signal')


# ---> OK!!
# 
# <br>
# 
# Apply the same process to all channels.

# In[ ]:


def create_signal_mod(train):
    left = 3641000
    right = 3829000
    thresh_dict = {
        3: [0.1, 2.0],
        2: [-1.1, 0.7],
        1: [-2.3, -0.6],
        0: [-3.8, -2],
    }
    
    train['signal_mod'] = train['signal'].values
    for ch in train[train['batch']==7]['open_channels'].unique():
        idxs_noisy = (train['open_channels']==ch) & (left<train.index) & (train.index<right)
        idxs_not_noisy = (train['open_channels']==ch) & ~idxs_noisy
        mean = train[idxs_not_noisy]['signal'].mean()

        idxs_outlier = idxs_noisy & (thresh_dict[ch][1]<train['signal'].values)
        train['signal_mod'][idxs_outlier]  = mean
        idxs_outlier = idxs_noisy & (train['signal'].values<thresh_dict[ch][0])
        train['signal_mod'][idxs_outlier]  = mean
    return train
df_tr = create_signal_mod(df_tr)


# In[ ]:


res = 100
x = np.arange(len(df_tr))

fig = plt.figure(figsize=(20, 5))
for i in df_tr[df_tr['batch']==7]['open_channels'].unique():
# for i in [0]:
    idxs = (df_tr['batch'] == 7) & (df_tr['open_channels'].values==i)
    plt.scatter(x[idxs], df_tr['signal_mod'].values[idxs], s=4, label=f'open_channel: {i}')
    
    
fig.legend(fontsize=10)
plt.title('batch7')
plt.ylabel('signal')


# <br>
# 
# plot per open_channels

# In[ ]:


colors = ['blue', 'orange', 'green', 'red']


# In[ ]:


res = 1
fig, axs = plt.subplots(4, 2, figsize=(25, 12))
fig.suptitle('batch7')
for i_ch, ch in enumerate(df_tr[df_tr['batch']==7]['open_channels'].unique()):
    # for i in [0]:
        idxs = (df_tr['batch'] == 7) & (df_tr['open_channels'].values==ch)
        axs[i_ch, 0].scatter(x[idxs][::res], df_tr['signal'].values[idxs][::res], s=4, label=f'open_channel: {ch}', color=colors[i_ch])
        axs[i_ch, 1].scatter(x[idxs][::res], df_tr['signal_mod'].values[idxs][::res], s=4, label=f'open_channel: {ch} (mod)', color=colors[i_ch])
        axs[i_ch, 0].legend(loc='upper left')
        axs[i_ch, 1].legend(loc='upper left')
        axs[i_ch, 0].set_ylabel('signal')
        axs[i_ch, 1].set_ylabel('signal_mod')
        axs[i_ch, 0].set_ylim(df_tr['signal'][idxs][::res].min(), df_tr['signal'][idxs][::res].max())
        axs[i_ch, 1].set_ylim(df_tr['signal'][idxs][::res].min(), df_tr['signal'][idxs][::res].max())


# --> goooood !!
