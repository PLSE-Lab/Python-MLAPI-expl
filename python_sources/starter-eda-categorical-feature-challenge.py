#!/usr/bin/env python
# coding: utf-8

# # Categorical Feature Challenge
# 
# This notebook provides some simple exploratory data analysis of the Categorical Feature Challenge data. Get ready for a lot of horizontal bar plots due to the nature of this data!
# 
# From the competition description: **This Playground competition will give you the opportunity to try different encoding schemes for different algorithms to compare how they perform. We encourage you to share what you find with the community.**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
import seaborn as sns


# ## Files
# - train.csv
# - test.csv
# - sample_submission.csv

# In[ ]:


get_ipython().system('ls ../input/cat-in-the-dat/ -GFlash --color')


# # Data
# From the data description:
# 
# ***
# In this competition, you will be predicting the probability [0, 1] of a binary target column.
# 
# The data contains binary features (bin_*), nominal features (nom_*), ordinal features (ord_*) as well as (potentially cyclical) day (of the week) and month features. The string ordinal features ord_{3-5} are lexically ordered according to string.ascii_letters.
# 
# Since the purpose of this competition is to explore various encoding strategies, the data has been simplified in that (1) there are no missing values, and (2) the test set does not contain any "unseen" feature values. (Of course, in real-world settings both of these factors are often important to consider!)
# ***

# In[ ]:


train = pd.read_csv('../input/cat-in-the-dat/train.csv')
test = pd.read_csv('../input/cat-in-the-dat/test.csv')
ss = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')


# ## Distribution of Target

# In[ ]:


target_0_pct = (train.groupby('target').count()['id'][0] / len(train) * 100)
target_0_count = train.groupby('target').count()['id'][0]
target_1_pct = (train.groupby('target').count()['id'][1] / len(train) * 100)
target_1_count = train.groupby('target').count()['id'][1]


# In[ ]:


print('Target of 0: in {} observations {:0.2f}%'.format(target_0_count, target_0_pct))
print('Target of 1: in {} observations {:0.2f}%'.format(target_1_count, target_1_pct))


# In[ ]:


train.groupby('target').count()['id'].plot(kind='barh', figsize=(15, 5), title='Target 1 vs 0')
plt.show()


# # bin Features
# - bin_0 to bin_2 : Binary Features (1 or 0)
# - bin_3 : T/F
# - bin_4 : Y/N
# 
# Regardless these are all binary features with only two possible values

# In[ ]:


train[['bin_0','bin_1','bin_2','bin_3','bin_4','target']].head()


# In[ ]:


bin_feats = ['bin_0','bin_1','bin_2','bin_3','bin_4']
for b in bin_feats:
    print(f'Feature {b} has unique values: {train[b].unique()}')


# In[ ]:


for f in bin_feats:
    fig, ax = plt.subplots(figsize=(15, 3))
    sns.catplot(y=f, hue="target", kind="count",
                palette="Greens_r", edgecolor=".6",
                data=train, ax=ax)
    ax.set_title(f'Distribution of Feature {f}')
    plt.close(2)
    plt.show()


# # nom Features

# In[ ]:


nom_feats = [f for f in train.columns if 'nom_' in f]


# In[ ]:


for n in nom_feats:
    if len(train[n].unique()) < 20:
        print(f'Feature {n} has unique values: {train[n].unique()}')
    else:
        print(f'Feature {n} has a lot of values: {len(train[n].unique())}')


# In[ ]:


for f in nom_feats[:5]:
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.catplot(y=f, hue="target", kind="count",
                palette="Blues_r", edgecolor=".6",
                data=train, ax=ax)
    ax.set_title(f'Distribution of Feature {f}')
    plt.close(2)
    plt.show()


# # ord features

# In[ ]:


ord_feats = [f for f in train.columns if 'ord_' in f]


# In[ ]:


for n in ord_feats:
    if len(train[n].unique()) < 200:
        print(f'Feature {n} has unique values: {train[n].unique()}')
    else:
        print(f'Feature {n} has a lot of values: {len(train[n].unique())}')


# In[ ]:


for f in ord_feats[:4]:
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.catplot(y=f, hue="target", kind="count",
                palette="Reds_r", edgecolor=".6",
                data=train, ax=ax)
    ax.set_title(f'Distribution of Feature {f}')
    plt.close(2)
    plt.show()


# # Time series features
# - Day
# - Month

# In[ ]:


ts_feats = ['day','month']
for f in ts_feats:
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.catplot(y=f, hue="target", kind="count",
                palette="Dark2_r", edgecolor=".6",
                data=train, ax=ax)
    ax.set_title(f'Distribution of Feature {f}')
    plt.close(2)
    plt.show()

