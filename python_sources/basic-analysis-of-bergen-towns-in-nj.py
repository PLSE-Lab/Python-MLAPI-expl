#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 30)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/covid19-new-jersey-nj-local-dataset/Covid-19-NJ-Bergen-Municipality.csv', index_col='Date', parse_dates=['Date'])
df


# In[ ]:


# Just fixing a bad data issue/typo
df.loc['2020-04-23', 'Elmwood Park'] = 407


# ### Let's find the (interesting) towns that have the highest number of cases

# In[ ]:


towns = set(df.columns) - set(['Total Presumptive Positives', 'New Daily cases'])
print("A few towns :")
list(towns)[:10]


# In[ ]:


highest_towns = df.iloc[-1][towns].nlargest(10).index
#Also, adding some towns that are of interest to me
print("Towns with highest number of cases + some towns of interest to me:")
highest_towns = list(highest_towns) + ['Edgewater', 'Fort Lee']
highest_towns


# ## Simple trendline of the cumulative positive cases in these towns

# In[ ]:


ax = df[highest_towns].plot(figsize=(20, 10), rot=45)
ax.set_ylabel('Number of positive cases')
ax.set_title('Total Positive cases over time by towns', fontweight='bold', fontsize='x-large')
ax.set_xticks(df[highest_towns].index)
ax.set_xticklabels(df[highest_towns].index.strftime('%b-%d'))
plt.show()


# Teaneck is definitely the leader in number of positive cases.

# ## Let's look at the rate of change between the towns

# In[ ]:


# Skipping initial numbers to focus more on the most recent trends/numbers
pct_change = df[40:].pct_change().rolling(10, 1).mean() * 100.0
ax = pct_change[highest_towns].plot(figsize=(20, 10), rot=45)
ax.set_ylabel('Percent daily rate of change')
ax.set_title('Percent daily rate of change over time by towns', fontweight='bold', fontsize='x-large')
ax.set_xticks(pct_change.index)
ax.set_xticklabels(pct_change.index.strftime('%b-%d'))
plt.show()


# The rate of new cases has decreased to <2% for all of these towns, which is great.

# ## Let's see the best & worst towns in terms of Rate of Change

# In[ ]:


# We'll look at data since March 20th to avoid the erratic data before.
# Also, we'll include only 10 towns that have the highest number of cases
recent_df = df[df.index > '2020-03-20']
recent_pct_change = recent_df.pct_change()
recent_pct_change[highest_towns].mean().sort_values()


# So, that's the best & worst towns (lower is better) based on the rate of change for positive cases since March 20th

# ## Days to double
# How many days does it take for the number of positive cases to double? For that, the rate of change needs to be 1.

# In[ ]:


# We'll take a Simple Moving Average to smoothen the irregularities with the raw numbers
days_to_double = 100.0/pct_change[15:]
ax = days_to_double[set(highest_towns) - {'Elmwood Park', 'Teaneck'}].rolling(5, 1).mean().plot(figsize=(20, 10), rot=45)
ax.set_ylabel('Number of days for cases to double')
ax.set_title('Number of days for cases to double over time by towns', fontweight='bold', fontsize='x-large')
ax.set_xticks(days_to_double.index)
ax.set_xticklabels(days_to_double.index.strftime('%b-%d'))
plt.show()


# Most of the towns seem to be on the right trend of an increasing number of days it takes to double the number of cases.
# 

# ## Logarithmic plot for New Cases vs Total cases
# We'll just plot for the towns that have the highest number of cases. Also, we'll take a Simple Moving average to smoothen the curve (remove irregularities).

# In[ ]:


melted_df = df[highest_towns].reset_index().melt(id_vars='Date', var_name='Town', value_name='Total Cases')
melted_df


# In[ ]:


melted_df['New Cases'] = melted_df.groupby('Town')['Total Cases'].transform(lambda x: x.diff())
melted_df


# In[ ]:


melted_df['New Cases'] = melted_df.groupby('Town')['New Cases'].transform(lambda x: x.rolling(8, 1).mean())


# In[ ]:


plt.figure(figsize=(20,20))
grid = sns.lineplot(x="Total Cases", y="New Cases", hue="Town", data=melted_df)
grid.set(xscale="log", yscale="log")
grid.set_title('New cases vs existing cases at log scale', fontweight='bold', fontsize='x-large')
plt.show()


# So, all towns with the highest number of cases are increasing at the 45-degree exponential-growth-rate line on this curve initially. But, all these towns are tapering off & their rate of daily cases is consistently decreasing & we can say that the curve for these towns has flattened. For more info, please see my inspiration here :
# 
# https://www.youtube.com/watch?v=54XLXg4fYsc
# 
# Thanks to minutephysitcs Youtube channel for creating the above video!

# **We'll plot the cases, rate of change & the number of days that it takes to double for some towns that are interesting to me for the last 15 days**. Feel free to fork the notebook & do the same for towns you are interested in.

# In[ ]:


towns = ['Cliffside Park', 'Edgewater', 'Englewood', 'Englewood Cliffs', 'Fort Lee', 'Hackensack', 'Leonia', 'Paramus', 'Palisades Park']
int_df = df[-15:][towns]


# In[ ]:


ax = int_df.plot(figsize=(20, 10), rot=45)
ax.set_ylabel('Number of positive cases')
ax.set_title('Total Positive cases over time by towns', fontweight='bold', fontsize='x-large')
ax.set_xticks(int_df[towns].index)
ax.set_xticklabels(int_df[towns].index.strftime('%b-%d'))
plt.show()


# In[ ]:


# Skipping initial numbers to focus more on the most recent trends/numbers
pct_change = int_df.pct_change().rolling(4, 1).mean() * 100.0
ax = pct_change.plot(figsize=(20, 10), rot=45)
ax.set_ylabel('Percent daily rate of change')
ax.set_title('Percent daily rate of change over time by towns', fontweight='bold', fontsize='x-large')
ax.set_xticks(pct_change.index)
ax.set_xticklabels(pct_change.index.strftime('%b-%d'))
plt.show()


# In[ ]:


# We'll take a Simple Moving Average to smoothen the irregularities with the raw numbers
days_to_double = 100.0/pct_change
ax = days_to_double.rolling(5, 1).mean().plot(figsize=(20, 10), rot=45)
ax.set_ylabel('Number of days for cases to double')
ax.set_title('Number of days for cases to double over time by towns', fontweight='bold', fontsize='x-large')
ax.set_xticks(days_to_double.index)
ax.set_xticklabels(days_to_double.index.strftime('%b-%d'))
plt.show()


# In[ ]:




