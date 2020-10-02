#!/usr/bin/env python
# coding: utf-8

# The team rating is given, but not the function that gets win probability from two ratings. 
# 
# We can only look at the probabilities for the first or second rounds since the rest are aggregates. The way around that would be simulation, but that's a lot of code to set up for the tourney structure. For now, let's just see if 538's win probabilities follow readily from the ratings.

# In[ ]:


import seaborn as sns
import numpy as np
from scipy import stats, optimize
import pandas as pd


# In[ ]:


df = pd.read_csv("../input/fivethirtyeight_ncaa_forecasts (2).csv")
df.head()


# In[ ]:


matchups = [[str(x+1), str(16-x)] for x in range(8)]
df = df[df.gender == 'mens']

pre = df[df.playin_flag==1]
data = []
for region in pre.team_region.unique():
    for seed in range(2, 17):
        res = pre[(pre.team_region == region) & (pre.team_seed.isin([str(seed)+'a', str(seed)+'b']))]
        if res.shape[0] > 1:
            data.append([])
            for _, row in res.iterrows():
                data[-1].extend([row.team_rating, row.rd1_win])

post = df[df.playin_flag == 0]
for region in post.team_region.unique():
    for matchup in matchups:
        res = post[(post.team_region == region) & (post.team_seed.isin(matchup))]
        if res.shape[0] > 1:
            data.append([])
            for _, row in res.iterrows():
                data[-1].extend([row.team_rating, row.rd2_win])


# In[ ]:


match = pd.DataFrame(data, columns=['Team1_Rating',"Team1_Prob", "Team2_Rating", "Team2_Prob"])


# In[ ]:


match['delta'] = match.Team1_Rating - match.Team2_Rating
match['win_extra'] = match.Team1_Prob - 0.5


# In[ ]:


sns.regplot('delta', 'win_extra', data=match, order=2);


# There are a couple of cases where having a higher rating means a lower win probability. 538 probably used Monte-Carlo to get the probabilities, but it's still surprising that a higher rating can mean a lower win probability. 
# 
# Their model that generated these probably also accounts for additional factors, like a 'home court' advantage due to where the game is, or other factors that aren't wrapped into the ratings.

# Assuming that every team has the same Normal distribution for its skill that the subtraction of yields the probability of winning, then the variance of the resulting distribution can be found and used to learn about the team's assumed skill distribution.

# In[ ]:


def matcher(std, diff, prob):
    p = stats.norm.cdf(0, diff, std)
    return np.abs(p-prob)

stds = []
for _, row in match.iterrows():
    x0 = 1
    res = optimize.minimize(matcher, x0=x0, args=(row.delta, row.Team1_Prob))
    while res.status != 0 or res.x == x0:
        x0 *= 5
        res = optimize.minimize(matcher, x0=x0, args=(row.delta, row.Team1_Prob))
        if x0 > 1000:
            break
    stds.append(res.x)


# In[ ]:


stds


# A look at those standard deviations shows what I assumed, that 538 has a lot more going on than the simplified constant standard deviation distribution of skill that you see in a lot of March Madness models. 

# In[ ]:




