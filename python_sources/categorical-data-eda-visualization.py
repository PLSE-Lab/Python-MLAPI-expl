#!/usr/bin/env python
# coding: utf-8

# ![](https://media.giphy.com/media/o0vwzuFwCGAFO/giphy.gif)
# 
# I participated in last competition so much, I will participate again.
# 
# First of all, I'm going to do EDA to come up with an idea of the overall distribution or idea of the data.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno 

import chart_studio.plotly as py
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os 

print(os.listdir('../input/cat-in-the-dat-ii'))


# In[ ]:


# matplotlib setting
plt.rc('font', size=12) 
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=12) 
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12) 
plt.rc('legend', fontsize=12) 
plt.rc('figure', titlesize=14) 
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))


# In[ ]:


train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
train.head()


# In[ ]:


target, train_id = train['target'], train['id']
test_id = test['id']
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)
print(train.shape)
print(test.shape)


# In[ ]:


print(train.columns)


# **feature list**
# 
# It's important to know what each feature is, because you need to check how you encode or distribute based on the feature.
# 
# - **bin 0~4** : Binary Feature, label encoding
# - **nom 0~9** : Nominal Feature
# - **ord 0~5** : Ordinal Feature
# - **day/month** : Date, cycle encoding 
# 

# ## Total Distribution
# 
# Let's first look at the overall distribution of the data.

# In[ ]:


msno.matrix(train)


# This data seems to have a lot of missing value unlike last time.
# 
# Let's look at the sorted values.

# In[ ]:


msno.matrix(train, sort='ascending')


# You can see that about half are empty.

# In[ ]:


null_rate = [train[i].isna().sum() / len(train) for i in train.columns]
fig, ax = plt.subplots(1,1,figsize=(20, 7))
sns.barplot(x=train.columns, y=null_rate, ax=ax,color='gray')
ax.set_title("Missing Value Rate (Train)")
ax.set_xticklabels(train.columns, rotation=40)
ax.axhline(y=0.03, color='red')
plt.show()


# In[ ]:


null_rate = [test[i].isna().sum() / len(train) for i in test.columns]
fig, ax = plt.subplots(1,1,figsize=(20, 7))
sns.barplot(x=test.columns, y=null_rate, ax=ax,color='gray')
ax.set_title("Missing Value Rate (Test)")
ax.set_xticklabels(test.columns, rotation=40)
ax.axhline(y=0.02, color='red')
plt.show()


# - The missing value (train) seems to make the data roughly 3%.
# - The missing value (test) seems to make the data roughly 2%.
# 
# What about the target value distribution?

# In[ ]:


target_dist = target.value_counts()

fig, ax = plt.subplots(1, 1, figsize=(8,5))

barplot = plt.bar(target_dist.index, target_dist, color = 'lightgreen', alpha = 0.8)
barplot[1].set_color('darkred')

ax.set_title('Target Distribution')
ax.annotate("percentage of target 1 : {}%".format(target.sum() / len(target)),
              xy=(0, 0),xycoords='axes fraction', 
              xytext=(0,-50), textcoords='offset points',
              va="top", ha="left", color='grey',
              bbox=dict(boxstyle='round', fc="w", ec='w'))

plt.xlabel('Target', fontsize = 12, weight = 'bold')
plt.show()


# First of all, you can see that the target ratio is unbalanced, rather than last data.

# ## Binary Feature
# 
# Let's start with the **binary feature.**

# In[ ]:


fig, ax = plt.subplots(1,5, figsize=(30, 8))
for i in range(5): 
    sns.countplot(f'bin_{i}', data= train, ax=ax[i])
    ax[i].set_ylim([0, 600000])
    ax[i].set_title(f'bin_{i}', fontsize=15)
fig.suptitle("Binary Feature Distribution (Train Data)", fontsize=20)
plt.show()


# In[ ]:


fig, ax = plt.subplots(1,5, figsize=(30, 8))
for i in range(5): 
    sns.countplot(f'bin_{i}', data= test, ax=ax[i], alpha=0.7,
                 order=test[f'bin_{i}'].value_counts().index)
    ax[i].set_ylim([0, 600000])
    ax[i].set_title(f'bin_{i}', fontsize=15)
fig.suptitle("Binary Feature Distribution (Test Data)", fontsize=20)
plt.show()


# The overall `binary feature` distribution between `train` and `test` seems to be similar.

# It can be seen that as $i$ of ${bin}_i$ increases, the distribution approaches 50%.

# In[ ]:


fig, ax = plt.subplots(1,5, figsize=(30, 8))
for i in range(5): 
    sns.countplot(f'bin_{i}', hue='target', data= train, ax=ax[i])
    ax[i].set_ylim([0, 500000])
    ax[i].set_title(f'bin_{i}', fontsize=15)
fig.suptitle("Binary Feature Distribution (Train Data)", fontsize=20)
plt.show()


# ## Nominal Feature
# 
# From nominal data, we need to look more closely at the distribution.

# In[ ]:


fig, ax = plt.subplots(2,3, figsize=(30, 15))
for i in range(5): 
    sns.countplot(f'nom_{i}', data= train, ax=ax[i//3][i%3],
                 order=train[f'nom_{i}'].value_counts().index)
    ax[i//3][i%3].set_ylim([0, 350000])
    ax[i//3][i%3].set_title(f'nom_{i}', fontsize=15)
fig.suptitle("Nominal Feature Distribution (Train Data)", fontsize=20)
plt.show()


# In[ ]:


fig, ax = plt.subplots(2,3, figsize=(30, 15))
for i in range(5): 
    sns.countplot(f'nom_{i}', data= test, ax=ax[i//3][i%3],
                 order=test[f'nom_{i}'].value_counts().index,
                 alpha=0.7)
    ax[i//3][i%3].set_ylim([0, 250000])
    ax[i//3][i%3].set_title(f'nom_{i}', fontsize=15)
fig.suptitle("Nominal Feature Distribution (Test Data)", fontsize=20)
plt.show()


# The overall `nominal feature` distribution between `train` and `test` seems to be similar.

# In[ ]:


fig, ax = plt.subplots(2,3, figsize=(30, 15))
for i in range(5): 
    sns.countplot(f'nom_{i}', hue='target', data= train, ax=ax[i//3][i%3],
                 order=train[f'nom_{i}'].value_counts().index)
    ax[i//3][i%3].set_ylim([0, 300000])
    ax[i//3][i%3].set_title(f'nom_{i}', fontsize=15)
fig.suptitle("Nominal Feature Distribution (Train Data)", fontsize=20)
plt.show()


# In[ ]:


for i in range(5):
    data = train[[f'nom_{i}', 'target']].groupby(f'nom_{i}')['target'].value_counts().unstack()
    data['rate'] = data[1]  / (data[0] + data[1] )
    data.sort_values(by=['rate'], inplace=True)
    display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39'))


# The visualization took too long and we looked at the rest of the features and found many unique elements:

# In[ ]:


train[[f'nom_{i}' for i in range(5, 10)]].describe(include='O')


# There seems to be something similar between `nom_7` and `nom_8`.

# In[ ]:


fig, ax = plt.subplots(2,1, figsize=(30, 10))
for i in range(7,9): 
    sns.countplot(f'nom_{i}', data= train, ax=ax[i-7],
                  order = train[f'nom_{i}'].dropna().value_counts().index)
    ax[i-7].set_ylim([0, 5500])
    ax[i-7].set_title(f'bin_{i}', fontsize=15)
    ax[i-7].set_xticks([])
fig.suptitle("Nominal Feature Distribution (Train Data)", fontsize=20)
plt.show()


# In[ ]:


fig, ax = plt.subplots(2,1, figsize=(30, 10))
for i in range(7,9): 
    sns.countplot(f'nom_{i}', hue='target', data= train, ax=ax[i-7],
                  order = train[f'nom_{i}'].dropna().value_counts().index)
    ax[i-7].set_ylim([0, 5000])
    ax[i-7].set_title(f'bin_{i}', fontsize=15)
    ax[i-7].set_xticks([])
fig.suptitle("Nominal Feature Distribution (Train Data)", fontsize=20)
plt.show()


# The comparison after sorting **does not seem** to have **high similarity**. (The distribution looks similar, but it's too different in detail.)
# 
# However, given that the numbers are the same and that the bending points on the graph are at similar points in the sort order by size, we assume that there is some preprocessing to see the relationship between the two features.

# ## Ordinal Feature

# In[ ]:


train[[f'ord_{i}' for i in range(6)]].describe(include='all')


# In[ ]:


fig, ax = plt.subplots(1,3, figsize=(30, 8))

ord_order = [
    [1.0, 2.0, 3.0],
    ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],
    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']
]

for i in range(3): 
    sns.countplot(f'ord_{i}', hue='target', data= train, ax=ax[i],
                  order = ord_order[i]
                 )
    ax[i].set_ylim([0, 200000])
    ax[i].set_title(f'ord_{i}', fontsize=15)
fig.suptitle("Ordinal Feature Distribution (Train Data)", fontsize=20)
plt.show()


# Oddly, it feels like the 3 graphs are gradually expanding, which may be useful to check again later with correlation.

# In[ ]:


fig, ax = plt.subplots(1,2, figsize=(24, 8))

for i in range(3, 5): 
    sns.countplot(f'ord_{i}', hue='target', data= train, ax=ax[i-3],
                  order = sorted(train[f'ord_{i}'].dropna().unique())
                 )
    ax[i-3].set_ylim([0, 75000])
    ax[i-3].set_title(f'ord_{i}', fontsize=15)
fig.suptitle("Ordinal Feature Distribution (Train Data 3~4)", fontsize=20)
plt.show()


# Oddly... it feels like the 2 graphs are gradually expanding...?
# 
# Using this part seems to minimize the feature.

# In[ ]:


for i in range(5):
    data = train[[f'ord_{i}', 'target']].groupby(f'ord_{i}')['target'].value_counts().unstack()
    data['rate'] = data[1]  / (data[0] + data[1] )
    data.sort_values(by=['rate'], inplace=True)
    display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39'))


# **I personally guess**
# 
# By sorting the values by the ratio of 1 in the target value, multiple ord features were ordered. This is thought to be intentional when the data is created, and data that is out of sync is considered to be an error from missing values.
# 
# ---
# 
# ord_6 has a large number of unique values, so let's sort them by size.

# In[ ]:


fig, ax = plt.subplots(2,1, figsize=(24, 16))

xlabels = train['ord_5'].dropna().value_counts().index

print(len(xlabels))

# just counting
sns.countplot('ord_5', data= train, ax=ax[0], order = xlabels )
ax[0].set_ylim([0, 12000])
ax[0].set_xticklabels(xlabels, rotation=90, rotation_mode="anchor", fontsize=7)

# with hue
sns.countplot('ord_5', hue='target', data= train, ax=ax[1], order = xlabels )
ax[1].set_ylim([0, 10000])
ax[1].set_xticklabels(xlabels, rotation=90, rotation_mode="anchor", fontsize=7)

fig.suptitle("Ordinal Feature Distribution (Train Data 5)", fontsize=20)
plt.show()


# ## Day & Month 

# In[ ]:


fig, ax = plt.subplots(2,1, figsize=(24, 16))

sns.countplot('day', hue='target', data= train, ax=ax[0])
ax[0].set_ylim([0, 100000])

sns.countplot('month', hue='target', data= train, ax=ax[1])
ax[1].set_ylim([0, 80000])

fig.suptitle("Day & Month Distribution", fontsize=20)
plt.show()


# Surprisingly, the graph feels like something else.
# 
# > Any advice would be appreciated if it was just me.

# In[ ]:


data = train[['day', 'target']].groupby('day')['target'].value_counts().unstack()
data['rate'] = data[1]  / (data[0] + data[1] )
data.sort_values(by=['rate'], inplace=True)
display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39'))

data = train[['month', 'target']].groupby('month')['target'].value_counts().unstack()
data['rate'] = data[1]  / (data[0] + data[1] )
data.sort_values(by=['rate'], inplace=True)
display(data.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39'))


# ---
# 
# ## With Categofical Data with Encoding
# 
# To get the correlation, let's do some basic encodings and get the correlation.

# ### Binary

# In[ ]:


get_ipython().run_cell_magic('time', '', "bin_encoding = {'F':0, 'T':1, 'N':0, 'Y':1}\ntrain['bin_3'] = train['bin_3'].map(bin_encoding)\ntrain['bin_4'] = train['bin_4'].map(bin_encoding)\n\ntest['bin_3'] = test['bin_3'].map(bin_encoding)\ntest['bin_4'] = test['bin_4'].map(bin_encoding)")


# ### Nominal
# 
# I'll go ahead and target based encoding to believe the relationship between nom_7 and nom_8.

# In[ ]:


get_ipython().run_cell_magic('time', '', "from category_encoders.target_encoder import TargetEncoder\n\nfor i in range(10):\n    label = TargetEncoder()\n    train[f'nom_{i}'] = label.fit_transform(train[f'nom_{i}'].fillna('NULL'), target)\n    test[f'nom_{i}'] = label.transform(test[f'nom_{i}'].fillna('NULL'))")


# ### Ordinal

# In[ ]:


get_ipython().run_cell_magic('time', '', "ord_order = [\n    [1.0, 2.0, 3.0],\n    ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],\n    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']\n]\n\nfor i in range(1, 3):\n    ord_order_dict = {i : j for j, i in enumerate(ord_order[i])}\n    train[f'ord_{i}'] = train[f'ord_{i}'].fillna('NULL').map(ord_order_dict)\n    test[f'ord_{i}'] = test[f'ord_{i}'].fillna('NULL').map(ord_order_dict)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(3, 6):\n    ord_order_dict = {i : j for j, i in enumerate(sorted(list(set(list(train[f'ord_{i}'].dropna().unique()) + list(test[f'ord_{i}'].dropna().unique())))))}\n    train[f'ord_{i}'] = train[f'ord_{i}'].fillna('NULL').map(ord_order_dict)\n    test[f'ord_{i}'] = test[f'ord_{i}'].fillna('NULL').map(ord_order_dict)")


# In[ ]:


train.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'f, ax = plt.subplots(1, 3, figsize=(45, 14))\nfor idx, tp in  enumerate([\'pearson\', \'kendall\', \'spearman\']) :\n    corr = train.fillna(-1).corr(tp)\n    mask = np.zeros_like(corr, dtype=np.bool)\n    mask[np.triu_indices_from(mask)] = True\n    cmap = sns.diverging_palette(220, 10, as_cmap=True)\n    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.2, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[idx])\n    ax[idx].set_title(f\'{tp} correlation viz\')\nplt.show()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'f, ax = plt.subplots(1, 3, figsize=(45, 14))\nfor idx, tp in  enumerate([\'pearson\', \'kendall\', \'spearman\']) :\n    corr = test.fillna(-1).corr(tp)\n    mask = np.zeros_like(corr, dtype=np.bool)\n    mask[np.triu_indices_from(mask)] = True\n    cmap = sns.diverging_palette(220, 10, as_cmap=True)\n    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax[idx])\n    ax[idx].set_title(f\'{tp} correlation viz (test)\')\nplt.show()')


# First of all, it depends on the encoding, but in the case of `ord`, sorting in order has some correlation with the **target value**.
# I think ord is definitely useful as a feature.
# 
# If you look at the correlation for other features and find the encoding, I think you will get good insights. (Or target based encoding methods would be nice.)
# 
# nom is a target based encoding, so I'll skip further thinking.
# 
# I don't know if the features are not correlated with each other. Please let me know in the comments if I made a mistake during the process.

# 
# ## TO BE CONTINUE ...
# 
# The kernel is still in progress.
