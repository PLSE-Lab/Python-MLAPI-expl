#!/usr/bin/env python
# coding: utf-8

# # Why I write this notebook?
# Idea comes from this wonderful notebooks [randomforest](https://www.kaggle.com/frankmollard/randomforest), but Frank use R, so I try use python to show his plot.
# 
# I think good understand the data is the key point for this competition, and good visulization helps.
# 
# # Thanks
# clean data use Chris's dataset and Markus's notebook [clean-removal-of-data-drift](https://www.kaggle.com/friedchips/clean-removal-of-data-drift).
# what's more, there are many wonderful notebooks use different models, I learned a lot from this competition, thanks all of you.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv')


# In[ ]:


group_size = 500_000

train['group'] = train.index//group_size


# In[ ]:


train.head()


# In[ ]:


train['mean_sig'] = train.groupby(['open_channels','group'])['signal'].transform('mean')


# # thanks
# - https://www.kaggle.com/sirishks/0-918-only-signal-no-model

# In[ ]:


colors = sns.color_palette("Paired")
len(colors)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'plt.scatter')


# In[ ]:


res = 100
x = np.arange(len(train))[::res]
y = train['open_channels'].values[::res]
mean_sig = train['mean_sig'].values[::res]
sig = train['signal'].values[::res]

plt.figure(figsize=(25,5))
for i in range(11):
    plt.scatter(x[y==i], mean_sig[y==i], 1, marker='o', color=colors[i])


# In[ ]:


res = 100
x = np.arange(len(train))[::res]
y = train['open_channels'].values[::res]
mean_sig = train['mean_sig'].values[::res]
sig = train['signal'].values[::res]

plt.figure(figsize=(25,5))
for i in range(11):
    plt.scatter(x[y==i], mean_sig[y==i], 1, marker='o', color=colors[i])
    plt.scatter(x[y==i], sig[y==i], 1,  alpha=0.5, label=f'channel_{i}',color=colors[i])

    


# In[ ]:


print(train.shape)


# In[ ]:


print(train['group'].unique())


# In[ ]:


plt.figure(figsize=(25,5))
res = 50
for i in range(10):
    df_group = train[train['group']==i]
    x = range(group_size*i, group_size*i+len(df_group))[::res]
    y = df_group['signal'].values[::res]
    plt.plot(x, y)


# I remove group 7, because group 7 has some noise signal.

# In[ ]:



train_sub = train[train['group']!=7]
plt.figure(figsize=(25,5))
x = np.array(range(0, len(train_sub)))[::10]
y = train_sub['open_channels'].values[::10]
sig = train_sub['signal'].values[::10]
for i in range(11):
    plt.scatter(x[y==i], sig[y==i], 1,  alpha=0.5, label=f'channel_{i}')
plt.legend();


# In[ ]:


res = 10
train_sub = train
plt.figure(figsize=(25,5))
x = np.array(range(0, len(train_sub)))[::res]
y = train_sub['open_channels'].values[::res]
sig = train_sub['signal'].values[::res]
for i in range(11):
    plt.scatter(x[y==i], sig[y==i], 1,  alpha=0.5, label=f'channel_{i}')
#     plt.plot(x[y==i], sig[y==i], 1,  alpha=0.5, label=f'channel_{i}')
    
for i in range(11):
    plt.plot([i*500_000,i*500_000], [-4,8], color='black')
    
for i in range(10):
    plt.text(i*500_000+250_000, 5, i)


# In[ ]:




