#!/usr/bin/env python
# coding: utf-8

# # Score Based Boxes Pool Analysis
# 
# ## Introduction
# The following kernel is some analysis for a pool that I entered. This pool is based on the ones-digit of the final score similar to the standard super bowl pool. There is a 10 x 10 grid where the winning team is one axis and the losing team is other axis. The one's digit of the final score is used to determine the winner. For example, if the final score is 74-68 and you own the box that is Winning Team 4 - Losing Team 8, then you are the winner for that game. There are winners for every game of the tournament. I wanted to see what the trends are and what boxes are the most valuable based on patterns in final scores historically.

# In[11]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.

import os
# Any results you write to the current directory are saved as output.


# ## Regular Season Data
# 
# First let's take a look at the patterns we see in the regular season data since this has the most games and it will be the best indicator of any trends:

# In[26]:


reg_df = pd.read_csv('../input/RegularSeasonCompactResults.csv')
print('There are %s games in this dataset' % reg_df.shape[0])


# So this data set has over 150 thousand games which is a decent sized dataset. We expect this to capture the best results on average compared to the tournament results

# In[24]:



reg_df['Wmod10'] = reg_df.WScore % 10
reg_df['Lmod10'] = reg_df.LScore % 10
reg_df['score'] = list(zip(reg_df['Wmod10'], reg_df['Lmod10']))
counts = reg_df['score'].value_counts()
percentages = counts / reg_df.shape[0]
print(percentages.head(5))


# We can see right away by looking at the top 5 scores that there are some pairs that are clearly much better than others. Considering there are 100 possibilities, anything over 0.01 has an edge and the top score ( Winning Team 1, Losing Team 8) is > 30% more frequent than that expected value.
# 
# Let's look at the full heatmap:

# In[25]:


w_score = [i[0] for i in percentages.index.tolist()]
l_score = [i[1] for i in percentages.index.tolist()]
percentages = percentages.tolist()

data = np.nan * np.empty((10, 10))
data[w_score, l_score] = percentages

f, ax = plt.subplots(figsize=(14, 8))
axs = sns.heatmap(data, annot=True, fmt='f', linewidths=0.25)
plt.title('Regular Season Trends')
plt.ylabel('Winning Team')
plt.xlabel('Losing Team')
plt.show()


# The first thing that pops out is the diagonal where the ones digits are even. This behavior is easily explained by the fact that a game cannot end in a tie. Therefore, the game must end with a difference of a positive multiple of 10 for this result to occur. The lowest of which is 10 itself which makes this a less likely scenario.
# 
# Now lets look at the NCAA Tournament games.

# ## NCAA Tournament Data
# Next, let's do the same with this data. I'm expecting this dataset to be a lot smaller.

# In[29]:


tourney_df = pd.read_csv('../input/NCAATourneyCompactResults.csv')
print('There are %s games in this dataset' % tourney_df.shape[0])


# The smaller sample size here (2117 games) is probably going to have a lot more noise. So let's take a look at the top 5:

# In[33]:


tourney_df['Wmod10'] = tourney_df.WScore % 10
tourney_df['Lmod10'] = tourney_df.LScore % 10
tourney_df['score'] = list(zip(tourney_df['Wmod10'], tourney_df['Lmod10']))
counts = tourney_df['score'].value_counts()
percentages = counts / tourney_df.shape[0]
print(percentages.head(5))


# So the top 5 here are way outside of what we see in the regular season data. Let's see what the heatmap looks like:

# In[34]:


w_score = [i[0] for i in percentages.index.tolist()]
l_score = [i[1] for i in percentages.index.tolist()]
percentages = percentages.tolist()

data = np.nan * np.empty((10, 10))
data[w_score, l_score] = percentages

f, ax = plt.subplots(figsize=(14, 8))
axs = sns.heatmap(data, annot=True, fmt='f', linewidths=0.25)
plt.title('NCAA Tournament Trends')
plt.ylabel('Winning Team')
plt.xlabel('Losing Team')
plt.show()


# So this confirms that the sample size here isn't large enough for the trends that we're seeing in the regular season data to exist.
# 
# Now let's look at the secondary tournament data.

# ## Secondary Tournament Data
# 
# Now let's do the same with the secondary tournament data:

# In[35]:


secondary_df = pd.read_csv('../input/SecondaryTourneyCompactResults.csv')
print('There are %s games in this dataset' % secondary_df.shape[0])


# With only 1484 games in this dataset, we can expect it to be similar to the NCAA tourney results

# In[36]:


secondary_df['Wmod10'] = secondary_df.WScore % 10
secondary_df['Lmod10'] = secondary_df.LScore % 10
secondary_df['score'] = list(zip(secondary_df['Wmod10'], secondary_df['Lmod10']))
counts = secondary_df['score'].value_counts()
percentages = counts / secondary_df.shape[0]
print(percentages.head(5))


# This looks similar to the NCAA tourney. Let's check the heatmap:

# In[37]:


w_score = [i[0] for i in percentages.index.tolist()]
l_score = [i[1] for i in percentages.index.tolist()]
percentages = percentages.tolist()

data = np.nan * np.empty((10, 10))
data[w_score, l_score] = percentages

f, ax = plt.subplots(figsize=(14, 8))
axs = sns.heatmap(data, annot=True, fmt='f', linewidths=0.25)
plt.title('Secondary Tournament Trends')
plt.ylabel('Winning Team')
plt.xlabel('Losing Team')
plt.show()


# Okay so again this is kind of spread across kind of randomly where the trends don't really appear like we see in the regular season data

# ## Combined Data
# 
# Now let's combine all of the data that we have so we can see the final results in a heatmap:

# In[42]:


total_df = pd.concat([reg_df, tourney_df, secondary_df])
counts = total_df['score'].value_counts()
percentages = counts / total_df.shape[0]
w_score = [i[0] for i in percentages.index.tolist()]
l_score = [i[1] for i in percentages.index.tolist()]
percentages = percentages.tolist()

data = np.nan * np.empty((10, 10))
data[w_score, l_score] = percentages

f, ax = plt.subplots(figsize=(14, 8))
axs = sns.heatmap(data, annot=True, fmt='f', linewidths=0.25)
plt.title('Combined Trends')
plt.ylabel('Winning Team')
plt.xlabel('Losing Team')
plt.show()


# So here are the top 10 scores to get in the form of (Winning Team, Losing Team)

# In[43]:


counts = total_df['score'].value_counts()
percentages = counts / total_df.shape[0]
print(percentages.head(10))


# ## Conclusion
# 
# So that completes the analysis. I was lucky enough to draw (0,8) as one of my numbers so hopefully this edge helps!
