#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.patches as patches

plt.style.use('seaborn')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
pd.set_option('max_columns', 100) # So we can see more columns

# Read in the training data
train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# In[ ]:


train


# In[ ]:


train.groupby('PlayId').first()['Yards'].plot(kind='hist',figsize=(15,5),bins=100)
plt.show()


# In[ ]:


train.groupby('PlayId').first()['Yards'].plot(kind='hist',figsize=(15,5),bins=100)
plt.show()


# In[ ]:


train.groupby('PlayId').first()['Yards']     .plot(kind = 'hist',
          figsize = (15,5),
          bins = 100,
          title = 'My Target is Yards ')
plt.show()


# In[ ]:


for i, d in train.groupby('Down'):
    print(i, d)


# In[ ]:


fig, axes = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
n = 0
for i, d in train.groupby('Down'):
    d['Yards'].plot(kind='hist',
                    bins=100,
                   color=color_pal[n],
                   ax=axes[n],
                   title=f'Yards Gained on down {i}')
    n+=1


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 5))
sns.violinplot(x='Distance-to-Gain',
               y='Yards',
               data=train.rename(columns={'Distance':'Distance-to-Gain'}),
               ax=ax)
plt.ylim(-10, 20)
plt.title('Yards vs Distance-to-Gain')
plt.show()


# In[ ]:


train.groupby('GameId')['PlayId']     .nunique()     .plot(kind='hist', figsize=(15, 5),
         title='Distribution of Plays per GameId',
         bins=50)
plt.show()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(data=train.groupby('PlayId').first()[['Distance','Down']],
            x='Down', y='Distance', ax=ax1)
ax1.set_title('Distance-to-Gain by Down')
sns.boxplot(data=train.groupby('PlayId').first()[['Yards','Down']],
            x='Down', y='Yards', ax=ax2)
ax2.set_title('Yards Gained by Down')
plt.show()


# In[ ]:


train['Distance'].plot(kind='hist',
                       title='Distribution of Distance to Go',
                       figsize=(15, 5),
                       bins=30,
                       color=color_pal[2])
plt.show()


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
train['S'].plot(kind='hist', ax=ax1,
                title='Distribution of Speed',
                bins=20,
                color=color_pal[0])
train['A'].plot(kind='hist',
                ax=ax2,
                title='Distribution of Acceleration',
                bins=20,
                color=color_pal[1])
train['Dis'].plot(kind='hist',
                  ax=ax3,
                  title='Distribution of Distance',
                  bins=20,
                  color=color_pal[2])
plt.show()


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
train.query("NflIdRusher == NflId")['S']     .plot(kind='hist',
          ax=ax1,
          title='Distribution of Speed (Ball Carrier Only)',
          bins=20,
          color=color_pal[0])
train.query("NflIdRusher == NflId")['A']     .plot(kind='hist',
          ax=ax2,
          title='Distribution of Acceleration (Ball Carrier Only)',
          bins=20,
          color=color_pal[1])
train.query("NflIdRusher == NflId")['Dis']     .plot(kind='hist',
          ax=ax3,
          title='Distribution of Distance (Ball Carrier Only)',
          bins=20,
          color=color_pal[2])
plt.show()


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sns.pairplot(train.query("NflIdRusher == NflId").sample(1000)[['S','Dis','A','Yards','DefensePersonnel']],
            hue='DefensePersonnel')
plt.suptitle('Speed, Acceleration, Distance traveled, and Yards Gained for Rushers')
plt.show()


# In[ ]:


train['SnapHandoffSeconds'] = (pd.to_datetime(train['TimeHandoff']) -                                pd.to_datetime(train['TimeSnap'])).dt.total_seconds()


# In[ ]:


train['SnapHandoffSeconds']


# In[ ]:


train['TimeSnap']


# In[ ]:


train.groupby('SnapHandoffSeconds').first()['Yards'].plot(kind='hist', bins=10)


# In[ ]:


train.groupby('SnapHandoffSeconds')['Yards'].mean().plot(kind='bar',
                                                         color=color_pal[1],
                                                         figsize=(15, 5),
                                                         title='Average Yards Gained by SnapHandoff Seconds')
plt.show()


# In[ ]:




