#!/usr/bin/env python
# coding: utf-8

# An exploration of the leaderboard (as of December 7th), just for fun.

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


DATA_PATH = '../input/santaleaderboard/traveling-santa-2018-prime-paths-publicleaderboard.csv'
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime( df['SubmissionDate'])


# In[ ]:


df.head()


# In[ ]:


def date_score_scatter(df, cutoff=1e9):
    d = df[df['Score'] <= cutoff]
    plt.figure(figsize=(12,8))
    plt.plot(d['Date'], d['Score'], '.')


# Let's start by seeing the overall distribution of scores over time.

# In[ ]:


date_score_scatter(df)


# There are clearly some patterns here, with at least 3 pretty distinct clusters of similar scores.
# 
# The topmost cluster looks to be at about 4.4e8, or 440 million. This is the score you get by just traversing the cities in ascending order. Most of these are probably from people sumbmitting the sample submission.
# 
# The cluster at around 200 million is probably from people sorting the points by x and/or y coordinates, as demonstrated [here](https://www.kaggle.com/seshadrikolluri/understanding-the-problem-and-some-sample-paths).
# 
# Let's zoom way in and look at the dense cluster at the bottom.

# In[ ]:


date_score_scatter(df, 2e6)


# This is the bottom 0.5% of the y-scale of the first plot. Zoomed in like this, we can see that there is some interesting structure here too.
# 
# The cluster at about 1.83 million is almost certainly from people doing a simple, greedy nearest-neighbor tour.
# 
# There are more clusters visible at the bottom, so let's zoom in again.

# In[ ]:


date_score_scatter(df, 1.54e6)


# OK, there is a large cluster around 1.533 million, with a bunch of submissions starting around November 21st and then petering out. This likely due to [this kernel](https://www.kaggle.com/wcukierski/concorde-solver). It seems like lots of people just ran pyconcorde and submitted the results.
# 
# The other dense cluster, starting November 28th, looks to be around 1.517 million. This is very likely due to [this kernel](https://www.kaggle.com/blacksix/concorde-for-5-hours), which demonstrated how to get that score by running concorde.
# 
# Let's zoom in one more time to check out the top contenders.

# In[ ]:


date_score_scatter(df, 1.52e6)


# I'm not sure how much I can read from the tea leaves here. I'm guessing that the slight downward trend of the dense cluster is due to people doing longer and longer runs of concorde, or finding slightly better prime-city optimizations on top of concorde's solutions. But that's just a guess.
# 
# At this point, let's take a closer look at how the top teams' scores have evolved over the competion so far.

# In[ ]:


def show_teams(df, cutoff):
    d = df[df['Score'] <= cutoff]
    plt.figure(figsize=(20,12))
    best = d[['TeamName','Score']].groupby('TeamName').min().sort_values('Score').index
    args = dict(data=d, x='Date', y='Score', hue='TeamName', hue_order=best, palette='muted')
    sns.lineplot(legend=False, **args)
    sns.scatterplot(legend=('brief' if len(best)<=30 else False), **args)


# In[ ]:


show_teams(df, 1517500)


# Well ain't that purdy.
# 
# One more zoom for the grand finale.

# In[ ]:


show_teams(df, 1516000)


# So much drama!
# 
# KMCoders and Farmers peeing further were neck-and-neck for a while. They were joined suddenly by KaizaburoChubachi, and it was looking like a three-horse race. But then, out of the blue, Prime Mover appeared and crushed them all, like Usain Bolt showing up at a high-school track meet. (I know, I just flipped my metaphor from horses to sprinters).
# 
# Surely it's as good as over - Prime Mover is untouchable. But wait! On December 6th, Farmers peeing further clearly had a breakthrough. They've made rapid improvements over the last couple of days, and could be on a trajectory to catch Prime Mover. Meanwhile, ultragamaza and Dodo Craze have both made rapid progress too, and now hold the 3rd and 4th spots
# 
# With over a month to go, it's still anybody's game.

# ### Update 2018-12-16

# In[ ]:


df2 = pd.read_csv('../input/leaderboard20181226/santa-leaderboard-2018-12-26.csv')
df2['Date'] = pd.to_datetime(df2['SubmissionDate'])


# In[ ]:


show_teams(df2, 1515750)


# ### Update 2019-01-09

# In[ ]:


df3 = pd.read_csv('../input/santaleaderboard20190109/santa-leaderboard-2019-01-09.csv')
df3['Date'] = pd.to_datetime(df3['SubmissionDate'])


# In[ ]:


show_teams(df3, 1515150)


# In[ ]:




