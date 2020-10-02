#!/usr/bin/env python
# coding: utf-8

# # EDA Animes Crunchyroll
# 
# I've created this notebook for two main reasons:
# 
# 1. Because this dataset is cool and i want to share my insights with you, feel free to give me ideas in the comments;
# 2. Because I always spend hours reading documentation to plot the same plots I've ploted over years and for some reason I just can't remember the syntax.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


# ## Read dataset

# In[ ]:


df_animes = pd.read_csv('../input/crunchyroll-anime-ratings/animes.csv')


# # EDA

# ## Create features
# 
# I'm only interested in animes with at least one vote, so I'll filter them here. Besides that, I want to create a few columns to help me in my analysis.

# In[ ]:


df_animes = df_animes[df_animes['votes'] > 0].copy()

# Get cols starting with genre_
genre_cols = [col for col in df_animes.columns if col.startswith('genre_')]

# Create columns
df_animes['votes_log'] = np.log10(df_animes['votes'])
df_animes['rate_rounded'] = round(df_animes['rate'])
df_animes['episodes_log'] = np.log10(df_animes['episodes'])
df_animes['qnt_genres'] = df_animes[genre_cols].sum(axis=1)


# ## Kernel Density Estimation
# 
# We have four plots here:
# 
# 1. Ratings seem high in my opinion, the majority of my animes are close to five rates;
# 2. Although seem like we don't have many voters, Crunchyroll probably should incentive people of voting more often;
# 3. Most animes seems to have between 20 and 40 episodes;
# 4. Most animes seems to have between 2 and 4 genres.

# In[ ]:


fig, axs  = plt.subplots(2, 2, figsize=(20, 10))

sns.kdeplot(df_animes['rate'], shade=True, ax=axs[0][0])
sns.kdeplot(df_animes['votes_log'], shade=True, color='green', ax=axs[0][1])
sns.kdeplot(df_animes[df_animes['episodes'] > 0]['episodes_log'], shade=True, ax=axs[1][0])
sns.kdeplot(df_animes['qnt_genres'], shade=True, color='green', ax=axs[1][1])

plt.show()


# ## Ratings

# ### Scatterplot
# 
# This plot tells us a lot! Since the ratings are an integer from zero to five, people don't vote between those values and we can see those vertical lines representing that. Also, the rounded shape pattern is caused because with fewer votes, more close to those intervals (1, 1.5, 2, 2.5 ...)

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 6))

ax = sns.scatterplot(x="rate", y="votes_log", hue='rate_rounded', data=df_animes)

plt.show()


# ## Joint distributions
# 
# This is plot shows us the same but with a different perspective, here we can see more clearly where is our interval denser

# In[ ]:


with sns.axes_style('white'):
    ax = sns.jointplot(x="rate", y="votes_log", data=df_animes, kind='hex', height=9)


# ## Genres

# In[ ]:


len(genre_cols)


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 6))

sns.kdeplot(df_animes[genre_cols].sum(), shade=True, color='green', ax=ax)

plt.show()


# ### Top 10 genres

# In[ ]:


top10_genres = df_animes[genre_cols].sum().sort_values(ascending=False).head(10)
top10_genres


# #### Rates per genre - top 10 genres
# 
# Looks like people doesn't have a clear preference of genre

# In[ ]:


sns.set_palette(sns.color_palette("Paired"))

fig, ax = plt.subplots(figsize=(15, 6))

for col in top10_genres.index:
    sns.kdeplot(df_animes[df_animes[col] == 1]['rate'], label=col, ax=ax)

plt.show()


# #### Votes per genre - top 10 genres

# In[ ]:


sns.set_palette(sns.color_palette("Paired"))

fig, ax = plt.subplots(figsize=(15, 6))

for col in top10_genres.index:
    sns.kdeplot(df_animes[df_animes[col] == 1]['votes_log'], label=col, ax=ax)

plt.show()


# ## Top 5 animes

# ### Number of votes
# 
# Looks like our data is missing the number of episodes for some animes, we should take care of it in our crawler

# In[ ]:


df_animes.sort_values('votes', ascending=False)[['anime', 'episodes', 'votes', 'rate']].head()


# ### Rating
# 
# I'll add a thrshold here to avoid animes with only one vote appear with 5 rating and force to have at least 2 episodes

# In[ ]:


df_animes[(df_animes['votes'] >= df_animes['votes'].median()) & (df_animes['episodes'] > 2)]    .sort_values('rate', ascending=False)[['anime', 'episodes', 'votes', 'rate']].head()


# ### Number of episodes

# In[ ]:


df_animes.sort_values('episodes', ascending=False)[['anime', 'episodes', 'votes', 'rate']].head()

