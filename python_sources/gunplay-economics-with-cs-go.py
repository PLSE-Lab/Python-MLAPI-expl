#!/usr/bin/env python
# coding: utf-8

# Counter Strike: Global Offensive is a first-person shooter video game in which two teams play as terrorists and counter-terrorists, respectively, and compete to blow up or defend an objective. Teams are five people per side, with players acquiring new weapons and items using points earned during the rounds in between rounds of play. Whichever team wins enough rounds, either by killing the entire enemy team or by successfully planting or defusing the bomb, wins the match.
# 
# Because of the way the game is designed, its playstyle is strongly influenced by economics: specifically, the economics of the point allocations. Buying better weapons is expensive, but also makes you likelier to win. What impact does this have on the way people play the game? I thought it would be interesting to find out!

# In[142]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
plt_kwargs = {'figsize': (10, 4)}


# In[143]:


weapon_events = pd.read_csv("../input/mm_master_demos.csv", index_col=0)
weapon_events['round_type'].value_counts()


# # Seconds
# 
# In CS:GO, each match consists of a certain number of rounds, with a time limit on the length of the rounds. The `seconds` field records how far into the total match time the weapon event occurred.

# In[144]:


fig = plt.figure(figsize=(10, 5))
sns.kdeplot(weapon_events['seconds'])
plt.suptitle("Weapon Event Time, Seconds into the Match")


# The main feature of this variable is that weapon events descend in frequency very strongly as the game's time limit winds down, probably because teams have won or quit by that time. On its own this feature isn't very informative, but it can be very interesting when taken in combination with other features.

# # Round Types
# 
# In CS:GO, each match consists of a certain number of rounds. The two teams start with the same number of points in the first round, and from there are provided additional points (dollars) at the beginning of each round: slightly more if the team won the previous round and slightly less if they lost.
# 
# These points make up the game's economy. Both teams must purchase weapons, armor, and consumables at the beginning of each round. The amount and quality of the equipment they can bring is limited by how many points they have available and are willing to spend.
# 
# There is a well-developed strategy to when and when not to spend points. On average, the better equipped the team, the better its chance of winning the round, but also the worse its cash reserves for future rounds. It's nevertheless obviously possible to beat a much better equipped team with worser weaponry, so sometimes (even often, in competitive games) teams will "bet" on a low equipment rollout and try to "steal" the round. Due to the way the game economy works, winning a round in this manner is highly cost-effective because it also sets your team up for winning future rounds.
# 
# The `round_types` column is a heuristical estimate of what type of round is being played at the time of event, based on the total/relative levels of spend on either team.
# 
# `NORMAL` rounds are just that. `ECO` and `SEMI_ECO` rounds are rounds in which one or more teams saves their points for future rounds. `FORCE_BUY` rounds are an extreme form when one team has exhausted its cash reserves and so is "handed" minimum-tier equipment (this is also part of game strategy). `PISTOL_ROUND` is an early game rounds where both teams are conserving points, and hence, using solely pistols (the cheapest kind of weapon).

# In[145]:


weapon_events['round_type'].value_counts().plot.bar(title='Round Types', **plt_kwargs)


# As you can see, teams play "normal" rounds almost less often than they play economically-driven ones! Managing the team's cash reserves is obviously a big part of the strategy.
# 
# We can see this in action through a different lense by looking at the raw data for this variable, `ct_eq_val` and `t_eq_val`, which measure the total weapon value buyed-in for either team.

# # Round Spend Value

# In[146]:


fig = plt.figure(figsize=(10, 5))
sns.kdeplot(weapon_events['ct_eq_val'].rename('Counter-Terrorists'))
sns.kdeplot(weapon_events['t_eq_val'].rename('Terrorists'))
plt.suptitle("Team Round Spend Values")


# As you can see teams have a very strong tendancy to buy equipment that's worth either "almost nothing" or "a lot". Interestingly, on buy rounds counter-terrorists tend to spend slightly more than Terrorists, possibly because they have a higher limit to the maximum amount of equipment they can buy.
# 
# How much advantage is created by additonal spend?

# In[147]:


match_level_data = weapon_events.groupby('file').head()


# In[153]:


df = pd.DataFrame().assign(winner=match_level_data['winner_side'], point_diff=match_level_data['ct_eq_val'] - match_level_data['t_eq_val'])
df = df.assign(point_diff=df.apply(lambda srs: srs.point_diff if srs.winner[0] == 'C' else -srs.point_diff, axis='columns'), winner=df.winner.map(lambda v: True if v[0] == 'C' else False))

df = (df
     .assign(point_diff_cat=pd.qcut(df.point_diff, 10))
     .groupby('point_diff_cat')
     .apply(lambda df: df.winner.sum() / len(df.winner))
)
df.index = df.index.values.map(lambda inv: inv.left + (inv.right - inv.left) / 2).astype(int)

fig = plt.figure(figsize=(10, 5))
df.plot.line()
plt.suptitle("Play Advantage Created by Additional Spend")
ax = plt.gca()
ax.axhline(0.5, color='black')
ax.set_ylim([0, 1])
ax.set_xlabel('Spend')
ax.set_ylabel('% Games Won')


# Surprisingly little! It seems that spending significantly more cash on a round than your opponent will only buy you, at most, a 10 percent greater chance of victory.
# 
# Here's another way of visualizing this effect. In the chart that follows, the blue line is the probability distribution for rounds won, while the red line is the probability distribution for all rounds. Winning teams do spend statistically significantly more than losing teams, but the effect is very small, at least in proportion to the size of the effect I expected.

# In[149]:


fig = plt.figure(figsize=(10, 5))

sns.kdeplot(match_level_data.query('winner_side == "CounterTerrorist"').pipe(lambda df: df.ct_eq_val - df.t_eq_val).rename('Winning Matches'))
sns.kdeplot(match_level_data.pipe(lambda df: df.ct_eq_val - df.t_eq_val).rename('All Matches'))

plt.suptitle("Team Weapon Values")


# # Spend Utilization
# 
# When a team decides it's going to spend money, what is it spending that money on anyway? To find out, let's look at the `wp_type` in our events averaged by total spend (`ct_eq_val` plus `t_eq_val`).

# In[163]:


g = sns.FacetGrid(weapon_events.assign(
    total_val=weapon_events['ct_eq_val'] + weapon_events['t_eq_val']
), col="wp_type", col_wrap=4)
g.map(sns.kdeplot, 'total_val')


# It seems that Rifles, Snipers, and Heavy Weapons are equally probabilistically likely to appear closest to the 50,000\$ "sweet spot". As CS:GO has five-person teams, this is \$10,000/person; a logical, and interesting, numerical amount of dollars to spend.
# 
# Pistols are an exception, for the obvious reason that they, being the cheapest weapon, are the weapon of choice for the aforementioned `ECO` rounds. It's interesting to see also that many players play games at mid-range spend, but still using pistols. These players are likely loading up on body armor and other equipment instead.
# 
# If you're not satisfied using a pistol, an SMG seem to be an economical weapon choice. SMGs appear in \$10,000/person games reasonably often, but also have a second peak at around \$3,000 or so. I'm not sure why the dip in between, however.
# 
# Again, we see how strong the "all-or-nothing" effect is. Teams seem to be very reluctant to spend a mid-range amount of cash.
