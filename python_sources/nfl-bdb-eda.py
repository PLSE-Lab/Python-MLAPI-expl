#!/usr/bin/env python
# coding: utf-8

# Below is a simple EDA of the NFL Big Data Bowl
# 
# Here are the [rules](https://operations.nfl.com/the-rules/2019-nfl-rulebook/) of the game
# 
# If you find this notebook helpful please upvote as it will motivate me to continue providing this material
# All comments notes and fixes are welcome.

# In[ ]:


import os

import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# In[ ]:


df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
print(df.shape)
df.head()


# In[ ]:


df.dtypes


# In[ ]:


(df.isna().sum() / df.shape[0]).nlargest(12)


# The dataframe is pretty full with WindDirection missing only ~16% of the data followed by WindSpeed with ~13%. Given that the dataframe is 500k just rows dropping the nan rows might be possible.

# View the field, code taken from this kernel: https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position

# In[ ]:


def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax


# In[ ]:


sample = '20181230154135'
fig, ax = create_football_field()
df.query(f"PlayId == {sample} and Team == 'away'").plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=30, legend='Away')
df.query(f"PlayId == {sample} and Team == 'home'").plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=30, legend='Home')
plt.title(f'Play # {sample}')
plt.legend()
plt.show()


# ### PlayID analysis

# In[ ]:


df['PlayId'].value_counts().describe()


# There are 23171 unique playIds and each have played 22 accounts

# The player that has traveled the most distance

# In[ ]:


df.groupby(['PlayId']).agg({'Dis': 'sum'})['Dis'].nlargest(20).plot(kind='bar', figsize=(20, 5))


# The player that is the fastest on average

# In[ ]:


df.groupby(['PlayId']).agg({'S': 'mean'})['S'].nlargest(20).plot(kind='bar', figsize=(20, 5))


# ## GameID analysis

# In[ ]:


df['GameId'].value_counts().nlargest(20).plot(kind='bar', figsize=(20, 5))


# In[ ]:


df['GameId'].value_counts().describe()


# 512 unique game with varying amount of plays

# # Numeric data

# In[ ]:


numeric_df = df.select_dtypes('number').drop(['GameId', 'PlayId', 'X', 'Y'], axis=1)
print(numeric_df.columns)
numeric_df.head()


# I drop GameID PlayerId and their (X) (Y) position as those columns might not have a lot too look at here

# In[ ]:


(numeric_df / numeric_df.max()).boxplot(figsize=(20, 5), rot=90)


# In[ ]:


(numeric_df / numeric_df.max()).boxplot(figsize=(20, 5), rot=90)


# (S)peed (A)cceleration and (Yards) have a lot of outliars

# In[ ]:


sns.distplot(numeric_df['Yards'])


# In[ ]:


sns.distplot(numeric_df['S'])


# In[ ]:


sns.distplot(numeric_df['A'])


# In[ ]:


corr = numeric_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Suprising how (S)peed isn't really correlated to anything. As expected (A)cceleration is strongly correlated to (S)peed since (A) is a measure of the change of (S)

# # Categorical data

# In[ ]:


cat_df = df.select_dtypes('object').drop(['TimeHandoff'], axis=1)
print(cat_df.columns)
cat_df.head()


# # To be continued
