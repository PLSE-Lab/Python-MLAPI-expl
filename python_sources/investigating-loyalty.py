#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()


# In[ ]:


data = pd.read_csv('../input/elo-merchant-eda/data_train.csv.gz')


# In[ ]:


data.head()


# In[ ]:


data = data.sort_values(by='first_active_month')


# In[ ]:


data.head()


# In[ ]:


print('The earliest date is {}, most recent date is {}.'.format(data.first_active_month.iloc[0], data.first_active_month.iloc[-1]))


# In[ ]:


data.target.describe()


# In[ ]:


target_stats = data.groupby('first_active_month').target.agg(['count','max','min','mean','median'])


# In[ ]:


target_stats.head()


# In[ ]:


fig, ax = plt.subplots()
ax.plot(target_stats.iloc[:,0])
ax.set_xlabel(xlabel='First Active Month')
ax.set_ylabel(ylabel='Frequency')
ax.xaxis.set_tick_params(rotation=90)
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 5))
ax.set_title('Frequency of target in chronological order')


# In[ ]:


fig, ax = plt.subplots()
ax.plot(target_stats.iloc[:,[1,2]])
ax.set_xlabel(xlabel='First Active Month')
ax.set_ylabel(ylabel='Target')
ax.xaxis.set_tick_params(rotation=90)
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 5))
ax.set_title('Max and min of target in chronological order')
ax.legend(['Max','Min'])


# From the above plot, we can see that in almost all the month, there will be a loyalty score of less than -30.

# In[ ]:


fig, ax = plt.subplots()
ax.plot(target_stats.iloc[:,[3]])
ax.set_xlabel(xlabel='First Active Month')
ax.set_ylabel(ylabel='Target')
ax.xaxis.set_tick_params(rotation=90)
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 5))
ax.set_title('Mean of target in chronological order')
ax.legend(['Mean'])


# In[ ]:


fig, ax = plt.subplots()
ax.plot(target_stats.iloc[:,[4]])
ax.set_xlabel(xlabel='First Active Month')
ax.set_ylabel(ylabel='Target')
ax.xaxis.set_tick_params(rotation=90)
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 5))
ax.set_title('Median of target in chronological order')
ax.legend(['Median'])


# Let's investigate these less than -30 loyalty scores.

# In[ ]:


less_than_thirty = data.loc[data.target < -30]
less_than_thirty.head()


# In[ ]:


print('There are {} records with loyalty score less than -30.'.format(less_than_thirty.shape[0]))


# In[ ]:


less_than_thirty.describe()


# In[ ]:


less_than_thirty.target.unique()


# There is only one value for this loyalty score.

# In[ ]:


target_stats_less_than_thirty = less_than_thirty.groupby('first_active_month').target.agg(['count'])
target_stats_less_than_thirty.head()


# In[ ]:


fig, ax = plt.subplots()
ax.plot(target_stats_less_than_thirty.iloc[:,0])
ax.set_xlabel(xlabel='First Active Month')
ax.set_ylabel(ylabel='Frequency')
ax.xaxis.set_tick_params(rotation=90)
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 5))
ax.set_title('Frequency of target in chronological order')


# The distribution seems to follow the general distribution of the whole training sample.

# In[ ]:


target_stats = target_stats.join(target_stats_less_than_thirty, rsuffix='_less_than_thirty')


# In[ ]:


target_stats = target_stats.fillna(0)


# In[ ]:


target_stats.head()


# In[ ]:


target_stats['percent_of_less_than_thirty'] = target_stats.iloc[:,-1]/target_stats.iloc[:,0]


# In[ ]:


target_stats.head()


# In[ ]:


target_stats.percent_of_less_than_thirty.describe()


# In[ ]:


fig, ax = plt.subplots()
ax.plot(target_stats.iloc[:,-1])
ax.set_xlabel(xlabel='First Active Month')
ax.set_ylabel(ylabel='Percentage')
ax.xaxis.set_tick_params(rotation=90)
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 5))
ax.set_title('% target which is < -30 in chronological order')


# It looks like the percentage of target that is less than -30 is quite stable, with a slight decrease across the months.

# In[ ]:


data['feature_combn'] = data.progress_apply(lambda x: str([x.feature_1,x.feature_2, x.feature_3]), axis=1)


# In[ ]:


print('There are {} feature combinations.'.format(len(data.feature_combn.unique())))


# In[ ]:


len(data.feature_1.unique())*len(data.feature_2.unique())*len(data.feature_3.unique())


# In[ ]:


pd.crosstab(index=data.feature_1, columns=data.feature_2)


# In[ ]:


pd.crosstab(index=data.feature_1, columns=data.feature_3)


# In[ ]:


pd.crosstab(index=data.feature_2, columns=data.feature_3)


# In[ ]:


feature_combn_target_stats = data.groupby('feature_combn').target.agg(['count','max','min','mean','median'])


# In[ ]:


feature_combn_target_stats


# From the table, we can see that regardless of feature combination, the min loyalty score is still -33.219.

# In[ ]:


data_test = pd.read_csv('../input/elo-merchant-eda/data_test.csv.gz')


# In[ ]:


data_test['feature_combn'] = data_test.progress_apply(lambda x: str([x.feature_1,x.feature_2, x.feature_3]), axis=1)


# In[ ]:


print('There are {} feature combinations.'.format(len(data_test.feature_combn.unique())))


# In[ ]:


pd.crosstab(index=data_test.feature_1, columns=data_test.feature_2)


# In[ ]:


pd.crosstab(index=data_test.feature_1, columns=data_test.feature_3)


# In[ ]:


pd.crosstab(index=data_test.feature_2, columns=data_test.feature_3)


# Test set display the same relationships for features.

# In[ ]:




