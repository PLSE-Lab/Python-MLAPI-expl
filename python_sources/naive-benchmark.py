#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

import os
print(os.listdir("../input"))


# ### Let's listen to an audio clip!

# In[2]:


import IPython.display as ipd
import wave


# In[3]:


train_curated_files = os.listdir('../input/freesound-audio-tagging-2019/train_curated')


# In[4]:


ipd.Audio('../input/freesound-audio-tagging-2019/train_curated/' + train_curated_files[0])


# ### Let's make a naive prediction!

# In[5]:


train_noisy = pd.read_csv(
    '../input/freesound-audio-tagging-2019/train_noisy.csv', index_col='fname')
train_curated = pd.read_csv(
    '../input/freesound-audio-tagging-2019/train_curated.csv', index_col='fname')
submission = pd.read_csv(
    '../input/freesound-audio-tagging-2019/sample_submission.csv', index_col='fname')

labels = submission.columns.tolist()


# In[6]:


for label in labels:
    train_noisy[label] = 0
    train_curated[label] = 0 


# In[7]:


for row in tqdm_notebook(train_noisy.index):
    row_labels = train_noisy.loc[row, 'labels'].split(',')
    for label in row_labels:
        train_noisy.loc[row, label] = 1

for row in tqdm_notebook(train_curated.index):
    row_labels = train_curated.loc[row, 'labels'].split(',')
    for label in row_labels:
        train_curated.loc[row, label] = 1
        
        
train_noisy['num_labels'] = train_noisy[labels].sum(axis=1)
train_curated['num_labels'] = train_curated[labels].sum(axis=1)


# In[8]:


label_count = train_noisy[labels].sum(axis=0) + train_curated[labels].sum(axis=0)
label_pred = label_count / label_count.sum()

submission.loc[:,:] = label_pred.values[:, None].ravel()


# In[9]:


submission.to_csv('submission.csv')

