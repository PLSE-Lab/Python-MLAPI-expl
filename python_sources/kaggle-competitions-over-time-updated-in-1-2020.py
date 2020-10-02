#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib
from matplotlib import pyplot as plt
import numpy as np # linear algebra
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


pd.set_option('max_columns', 100)
plt.style.use('fivethirtyeight')
matplotlib.rcParams['figure.figsize'] = [24, 12]


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_dir = '/kaggle/input/meta-kaggle'
df_comp = pd.read_csv(f'{data_dir}/Competitions.csv')
df_team = pd.read_csv(f'{data_dir}/Teams.csv')
print(df_comp.shape, df_team.shape)


# In[ ]:


df_comp.head()


# In[ ]:


df_comp['DeadlineDate'] = pd.to_datetime(df_comp['DeadlineDate'])
print(f'{df_comp.DeadlineDate.min()} - {df_comp.DeadlineDate.max()}')


# # Competition Types

# In[ ]:


df_comp['HostSegmentTitle'].value_counts()


# ## Prizes

# In[ ]:


df_comp['RewardQuantity'].describe()


# In[ ]:


df_comp['RewardQuantity'].sum()


# ## Top 10 Competitions with Most Total Prize

# In[ ]:


df_comp.sort_values('RewardQuantity', ascending=False).head(10)


# # Number of Teams

# In[ ]:


df_comp['TotalTeams'].describe()


# In[ ]:


sum(df_comp['TotalTeams'] > 10)


# # Summary Plot

# In[ ]:


ax = sns.scatterplot(x='DeadlineDate',
                     y='TotalTeams',
                     hue='HostSegmentTitle',
                     size='RewardQuantity',
                     sizes=(100, 1000),
                     data=df_comp)
ax.set_xlim(df_comp.DeadlineDate.min(), df_comp.DeadlineDate.max())


# The same chart in the log-y scale.

# In[ ]:


ax = sns.scatterplot(x='DeadlineDate',
                     y='TotalTeams',
                     hue='HostSegmentTitle',
                     size='RewardQuantity',
                     sizes=(100, 1000),
                     data=df_comp)
ax.set_xlim(df_comp.DeadlineDate.min(), df_comp.DeadlineDate.max())
ax.set_yscale('log')
ax.set_ylim(10, 15000)


# In[ ]:




