#!/usr/bin/env python
# coding: utf-8

# # Introduction to The Box Office Prediction Dataset

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')
get_ipython().system('ls ../input/')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
ss = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


print('Train shape {}'.format(train.shape))
print('Test shape {}'.format(test.shape))


# ## Target Variable Distribution

# In[ ]:


train['revenue'].plot(kind='hist',
                      figsize=(15, 5),
                      bins=50,
                      title='Distribution of Revenue (Train Set)')
plt.show()
print('Revenue has mean {:1.0f} and standard deviation {:1.0f}'.format(train['revenue'].mean(), train['revenue'].std())) 


# Log transforms are a common way to deal with features or targets that are heavily skewed. Log transforms are also easy to interperet. For every increase of 1 in the log transform, we can say the revenue increased 10x. With the log transform we can see we've reduced the skew.

# In[ ]:


train['revenue_log'] = train['revenue'].apply(np.log)
train['revenue_log'].plot(kind='hist',
                      figsize=(15, 5),
                      bins=50,
                      title='Distribution of Log Revenue (Train Set)')
plt.show()


# ## Overview of Features

# In[ ]:


train['budget'].plot(kind='hist',
                      figsize=(15, 5),
                      bins=50,
                      title='Distribution of Budget (Train Set)',
                      color='blue')
plt.show()

# Use the log1p transform since some values are zero
train['budget_log'] = train['budget'].apply(np.log)
train['budget_log'] = train['budget_log'].replace(-np.inf, 0)
train['budget_log'].plot(kind='hist',
                      figsize=(15, 5),
                      bins=50,
                      title='Distribution of Log+1 Budget (Train Set)',
                      color='blue')
plt.show()


# ## Budget vs. Revenue

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sns.jointplot('budget_log', 'revenue_log', train.loc[train['budget_log'] > 1], kind='reg')
plt.show()


# ## genres
# This column contains a list, with a dictionary of the genere. We should be able to convert this into dummy variable columns.

# In[ ]:


# Thanks to @kamalchhirang for this kernel for this code: https://www.kaggle.com/kamalchhirang/eda-simple-feature-engineering-external-data
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d
train = train
train['genres_split'] = train['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
genres = train.genres_split.str.get_dummies(sep=',')
train = pd.concat([train, genres], axis=1, sort=False)


# Plot the distributions of values for the major genres (ones with at least 500 movies)

# In[ ]:


genre_list = genres.columns.values
for genre in genre_list:
    if len(train.loc[train[genre] == 1]) > 500:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        train.loc[train[genre] == 1]['budget_log'].plot(kind='hist', figsize=(15, 2), bins=50, title='{} Log Budget'.format(genre), ax=ax1, xlim=(0, 25))
        train.loc[train[genre] == 1]['revenue_log'].plot(kind='hist', figsize=(15, 2), bins=50, title='{} Log Revenue'.format(genre), ax=ax2, xlim=(0, 25))
        train.loc[train[genre] == 1].plot(x='budget_log', y='revenue_log', kind='scatter', ax=ax3)
        plt.show()


# In[ ]:


# Average values by genre

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 15))

color1 = list(plt.rcParams['axes.prop_cycle'])[1]['color']
genre_popularity = {}
for genre in genre_list:
    genre_popularity[genre] = train.loc[train[genre] == 1]['popularity'].mean()
pd.DataFrame(genre_popularity, index=['Average Popularity'])     .T.sort_values('Average Popularity')     .plot(kind='barh', color=color1, title='Average Popularity by Genre', ax=ax1, legend=False)

# Find the popularity of each genre
color2 = list(plt.rcParams['axes.prop_cycle'])[2]['color']
genre_budget = {}
for genre in genre_list:
    genre_budget[genre] = train.loc[train[genre] == 1]['budget'].mean()
pd.DataFrame(genre_budget, index=['Average Budget'])     .T.sort_values('Average Budget')     .plot(kind='barh', color=color2, title='Average Budget by Genre', ax=ax2, legend=False)

color4 = list(plt.rcParams['axes.prop_cycle'])[4]['color']
genre_revenue = {}
for genre in genre_list:
    genre_revenue[genre] = train.loc[train[genre] == 1]['revenue'].mean()
pd.DataFrame(genre_revenue, index=['Average Revenue'])     .T.sort_values('Average Revenue')     .plot(kind='barh', color=color4, title='Average Revenue by Genre', ax=ax3, legend=False)
plt.show()

