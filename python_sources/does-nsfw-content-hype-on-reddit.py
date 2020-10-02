#!/usr/bin/env python
# coding: utf-8

# # NSFW (18+) content On Reddit
# * ### How big is NSFW content?
# * ### How do people react on NSFW conent?
# * ### How many NSFW posts are deleted and who is deleting them?
# * ### Does the score rises once the post gets attention?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
reddit_data = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv')
print(reddit_data.head())


# In[ ]:


reddit_data.info()


# In[ ]:


reddit_data.isnull().sum() / len(reddit_data) * 100


# As we can see, the data is missing only for some variables

# In[ ]:


fig, ax = plt.subplots()
sns.countplot(reddit_data.over_18)
ax.set(xlabel='Adult content', title='Distribution of Conetent')


# The distrubution is extremely uneven so we have to look at the exact numbers 

# In[ ]:


( reddit_data[reddit_data['over_18'] == True]['id'].count() / reddit_data.shape[0] ) * 100


# NSFW content on reddit is only half a percent of all reddits posts.

# In[ ]:


fig, (ax0, ax1) = plt.subplots(
nrows=1,ncols=2, sharey=True, figsize=(14,4))
sns.kdeplot(reddit_data[reddit_data['over_18'] == True]['score'], ax=ax0, color='orange', label='Score')
sns.kdeplot(reddit_data[reddit_data['over_18'] == False]['score'], ax=ax1, color='blue', label='Score')
ax1.set(xlim=(0,1000), xlabel='Score', title='Mass-Oritnted Content')
ax0.set(xlabel='Score', title='NSFW Content', ylabel='Frequency')
ax0.axvline(x=737.9, label='Average ', linestyle='--', color='red')
ax1.axvline(x=197.2, label='Average', linestyle='--', color='red')
ax1.legend()
ax0.legend()


# In[ ]:


print(reddit_data[reddit_data['over_18'] == True]['score'].mean())
print(reddit_data[reddit_data['over_18'] == False]['score'].mean())


# Despite the fact that NSFW content pops up only once in two hundered posts, the average score of NSFW content is alsmost 4 time higer 738 to 197.

# In[ ]:


fig, (ax0, ax1) = plt.subplots(
nrows=1,ncols=2, sharey=True, figsize=(14,4))
sns.kdeplot(reddit_data[reddit_data['over_18'] == True]['num_comments'], ax=ax0, color='orange', label='Number of Comments')
sns.kdeplot(reddit_data[reddit_data['over_18'] == False]['num_comments'], ax=ax1, color='blue', label='Number of Comments')

ax1.set(xlim=(0,1000), xlabel='Number of Comments', title='Mass-Oritnted Content')
ax0.set(xlabel='Number of Comments', title='NSFW Content', ylabel='Frequency')
ax0.axvline(x=104.7, label='Average ', linestyle='--', color='red')
ax1.axvline(x=24.7, label='Average', linestyle='--', color='red')
ax1.legend()
ax0.legend()


# In[ ]:


print(reddit_data[reddit_data['over_18'] == True]['num_comments'].mean())
print(reddit_data[reddit_data['over_18'] == False]['num_comments'].mean())


# Similarly to score, on average, number of comments tends to be 4 times higer for NSWF content. 

# Does number of comments help get a higher score?

# In[ ]:


ax = sns.regplot(x='num_comments',y='score', data=reddit_data[reddit_data['over_18'] == True])
ax.set(xlabel='Number of Comments', ylabel='Score', title='Number of Comments vs Score')


# In[ ]:


fig, ax = plt.subplots()
sns.residplot(x='num_comments',y='score', data=reddit_data[reddit_data['over_18'] == True])
ax.set(xlabel='Number of Comments', ylabel='Score', title='Residual plot')


# Since residuals are not random and tend to cluster there is no correlation between number of comments and score of NSFW post

# How many NSWF posts get deleted?

# In[ ]:


(reddit_data[reddit_data['over_18'] == True]['removed_by'].count() / reddit_data[reddit_data['over_18'] == True]['removed_by'].ffill(0).count()) * 100


# Apparantly, only 2% of all NSFW comments get deleted 

# Who is responsible for filtereing NSFW vs Mass Orinited content?

# In[ ]:


fig, (ax0, ax1) = plt.subplots(
nrows=1,ncols=2, figsize=(14,4))
sns.countplot(reddit_data[reddit_data['over_18'] == True]['removed_by'].dropna(), ax=ax0)
sns.countplot(reddit_data[reddit_data['over_18'] == False]['removed_by'].dropna(), ax=ax1)
ax0.set(xlabel='Removed by',title='NSFW content', ylim=(0,20), ylabel='Count')
ax1.set(xlabel='Removed by',title='Mass Orinted content', ylabel='Count')


# In general,any content on Reddit is heavily filtered by moderedotrs. To my surprisde, ML algorithms are more likely to filter mass orinited than NSFW content.

# NSFW content takes up a very small part of Reddit. Nevertheless, NSFW content definitely attracts a dicent portion of attention from the community with both number of comments and score 4x times higher than average for mass orinted content. 
