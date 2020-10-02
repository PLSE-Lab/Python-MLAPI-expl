#!/usr/bin/env python
# coding: utf-8

# # PUBG Exploratory Data Analysis by Player, Group and Match (Ongoing)
# 
# PlayerUnknown's BattleGrounds (PUBG) it's a very popular battle royale game (even [Deadmau5](https://www.youtube.com/watch?v=VuKtfEFa9W0) likes it). With this EDA I'm going to perform an analysis of most common patterns in different matches, teams, and players. The main idea is to see what insights can we get.
# 
# Let's start importing the libraries and the dataset.

# In[ ]:


# Import libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#Create Dataframe
df = pd.read_csv('../input/train.csv')

#Look data
df.head(5)


# ## Explanation of Variables and First Hypotheses.
# The dataset gives to us these variables:
# 
# 1. **DBNOs** - Number of enemy players knocked.
# 2. **assists** - Number of enemy players this player damaged that were killed by teammates.
# 3. **boosts** - Number of boost items used.
# 4. **damageDealt** - Total damage dealt. Note: Self inflicted damage is subtracted.
# 5. **headshotKills** - Number of enemy players killed with headshots.
# 6. **heals** - Number of healing items used.
# 7. **killPlace** - Ranking in match of number of enemy players killed.
# 8. **killPoints** - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.)
# 9. **killStreaks** - Max number of enemy players killed in a short amount of time.
# 10. **kills** - Number of enemy players killed.
# 11. **longestKill** - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
# 12. **matchId** - Integer ID to identify match. There are no matches that are in both the training and testing set.
# 13. **revives** - Number of times this player revived teammates.
# 14. **rideDistance** - Total distance traveled in vehicles measured in meters.
# 15. **roadKills** - Number of kills while in a vehicle.
# 16. **swimDistance** - Total distance traveled by swimming measured in meters.
# 17. **teamKills** - Number of times this player killed a teammate.
# 18. **vehicleDestroys** - Number of vehicles destroyed.
# 19. **walkDistance** - Total distance traveled on foot measured in meters.
# 20. **weaponsAcquired** - Number of weapons picked up.
# 21. **winPoints** - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.)
# 22. **groupId** - Integer ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# 23. **numGroups** - Number of groups we have data for in the match.
# 24. **maxPlace** - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# 26. **winPlacePerc** - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.
# 
# Initial hypotheses/questions to solve with data:
# 1. More boosts used is related to a higher final position.
# 2. More heals used is related to a higher final position.
# 3. Is camper style effective?
# 4. Teamwork is related to a higher final position.
# 5. Are vehicles for run away or for easy kills?
# 

# ## More Boosts Used is Related to a Higher Final Position
# **Hypothesis:** Since boosts give one player benefits over the others, the more a player uses any boost, the more likely he is to win mini fights against others. This translates into a higher final position.
# 
# Let's perform a scatter plot to test this statement.

# In[ ]:


fig = plt.figure()
axis = fig.add_subplot(1,1,1)
axis.scatter(df['boosts'],df['winPlacePerc'])
axis.set(title='Win Place Percentile by Boost Used', xlabel='Boosts Used', ylabel='Win Place Percentile')
plt.show()


# Looking at the resulting graph we can say that under the 6 power-ups used there is no effect on the final position, but above that, the possibilities start to increase. 
# 
# An explanation for this pattern might be that the boosts are not really that powerful. Instead, the benefits come from those players who know how to use it. For predictions, the number of power-ups used in the match could be important.
# 
# Let's take a look at the boost patterns for top 10% only.
# 

# In[ ]:


winners = df[df['winPlacePerc'] >= 0.9]
winners['boosts'].describe()


# The final conclusions then are:
# 1. Boosts can tell about the final position of a player but mainly due player's ability.
# 1.  The top 10% players use 3.2 boosts in average each game.
# 1. The minimum number of boosts used for the top 10% is 0 (this means that boosts aren't a must to reach a high final position).

# ## More Heals Used is Related to a Higher Final Position
# **Hypothesis:** If two players are trying to kill each other, the one that finally wins needs more healing. So more heals means a higher final position.
# 
# For this test, it is necessary to take into account the "carried" factor, those players that achieve high final positions due to work of other players in the team.  To solve this problem a transformation of the data is needed, each row need to be a team rather than a single player, then filtering by the size of the team it's possible to get the alone players.
# 
# ![tranformation_example](https://i.imgur.com/uNeZjAx.jpg)
# 
# Let's perform this trasnformation.

# In[ ]:


# Transform data for teams.
df_groups = (df.groupby('groupId', as_index=False).agg({'Id':'count', 'matchId':'mean', 'assists':'sum', 'boosts':'sum',
                                'damageDealt':'sum', 'DBNOs':'sum', 'headshotKills':'sum',
                                'heals':'sum', 'killPlace':'mean', 'killPoints':'max', 'kills':'sum',
                                'killStreaks':'mean', 'longestKill':'mean', 'maxPlace':'mean', 'numGroups':'mean',
                                'revives':'sum', 'rideDistance':'max', 'roadKills':'sum', 'swimDistance':'max',
                                'teamKills':'sum', 'vehicleDestroys':'sum', 'walkDistance':'max',
                                'weaponsAcquired':'sum','winPoints':'max', 'winPlacePerc':'mean'}).rename(columns={'Id':'teamSize'}).reset_index())
# Show changes
df_groups.head(5)


# In order to clarify the transformations, the reference is this table:
# ![tranformation_rules](https://i.imgur.com/4yy1V0g.png)
# Basically count each player's id in the team (this will be the teamSize), sum for features related to "teamwork", use mean for features that has a scale (e.g. maxPlace) or is the same for all players in the team (e.g. winPlacePerc) and use maximun value for those features where the reference is the best player in the team (e.g. walkDistance).
# 
# Taking all of the above into account, now filtering by teamSize it's possible to solve the initial hypothesis.

# In[ ]:


# Get teams of size = 1
alone_players = df_groups[df_groups['teamSize'] == 1]

# Plot win place percentile by heals used
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(alone_players['heals'], alone_players['winPlacePerc'])
ax.set(title='Win Place Percentile by Heals Used', xlabel='Heals Used', ylabel='Win Place Percentile')
plt.show()


# The resulting plot confirms the idea that more heals tends to mean a higher final position. In this case, it's harder to believe that someone can win without any heals, what I think it's happening here is due custom matches.
# 
# In custom matches one can create a type of match where only kills matter for example. In this special situations it's normal to get winners that did'nt use heals.
# 
# Regardless, for prediction purposes heals could be significant, let's analyze the patterns for top 10% alone players.

# In[ ]:


# Top 10% alone players (at least 4 heals during match)
alone_winners = alone_players[alone_players['winPlacePerc'] >= 0.9]
alone_winners = alone_winners[alone_winners['heals'] > 3]
# Describe patterns
alone_winners['heals'].describe()


# With a mean of 6.7 heals per match, the conclusion is that heals could be a relevant feature for prediction.

# ## Is Camper Style Effective?
# **Hypothesis:** PUBG players must travel (walking, swimming or riding) to stay in the 'safe zone' and win. So, a "camper" play style isn't very effective. 
# 
# Let's perform a scatter plot to get the influence of these three featrues.

# In[ ]:


# Walk Distance
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df_groups['walkDistance'], df_groups['winPlacePerc'])
ax.set(title='Win Place Percentile by Walk Distance', xlabel='Walk Distance (m)', ylabel='Win Place Percentile')
plt.show()


# Although the graph shows that in general terms, the more meters a player walks, the higher his final position tends to be, there are also records of winning players with very little distance traveled.
# 
# We can't talk about a "camper" game style but surely 'walkDistance' is a feature that could be important to predict 'winPlacePerc'.

# In[ ]:


# Swim Distance
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df_groups['swimDistance'], df_groups['winPlacePerc'])
ax.set(title='Win Place Percentile by Swim Distance', xlabel='Swim Distance (m)', ylabel='Win Place Percentile')
plt.show()


# The very sparse results tells that there is no relation between the swim distance and the final win place. In general players tends to swim just a little more than a kilometer and this isn't really a "camper" style since maps don't tend to be very aquatic.
# 
# In conclusion, 'swimDistance' isn't really very significant to 'winPlacePerc' and probably could be excluded in the predicting part.

# In[ ]:


# Ride Distance
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df_groups['rideDistance'], df_groups['winPlacePerc'])
ax.set(title='Win Place Percentile by Ride Distance', xlabel='Ride Distance (m)', ylabel='Win Place Percentile')
plt.show()


# The graph shows a very shy trend between these two features, this tell us that vehicles are helpful but not necessary to win. Also, in general, players tends to rid just a little more than a kilometer and a half so it doesn't seem to be the case of players riding a lot just to hide and win ('camper' playstyle).
# 
# According to the 3 distances measured we are likely to say that be a camper in PUBG isn't really effective but is not completely clear.

# ## Teamwork is Related to a Higher Final Position
# **Hypothesis:** "Two heads are better than one", teams of average players perform better than one expert and three mediocre players.
# 
# To test this let's take a look of how the assists and revives influence in the final place and compare the pattern of top 10% teams with the rests.

# In[ ]:


# assists
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df_groups['assists'], df_groups['winPlacePerc'])
ax.set(title='Win Place Percentile by Total Group Assists', xlabel='Total Group Assists', ylabel='Win Place Percentile')
plt.show()


# Looking at the graph, seems like there is no relatin between the two features, let's see what happens to 'revives'.

# In[ ]:


# revives
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df_groups['revives'], df_groups['winPlacePerc'])
ax.set(title='Win Place Percentile by Total Group Revives', xlabel='Total Group Revives', ylabel='Win Place Percentile')
plt.show()


# In conclusion, neither assists nor revives have influences in the final place. Teamwork playstyle can't be judged just by how many assists and revives a team has, but actually, the plots show that keeping all the team alive (revives) or attacking in groups (assists) aren't that crucial to winning a PUBG match.
# 
# However, let's compare patterns between the top 10% players and the rest.

# In[ ]:


print('Top 10%')
print(winners['assists'].describe())
print('\nOther Players')
print(df[df['winPlacePerc'] < 0.9]['assists'].describe())


# The numbers reveal that most winners neither even assists. However, the mean is bigger than the other players (0.79 vs 0.02) and can be a sign that final match moments is where teamwork really make a difference.
# 
# Let's analyze revives.

# In[ ]:


print('Top 10%')
print(winners['revives'].describe())
print('\nOther Players')
print(df[df['winPlacePerc'] < 0.9]['revives'].describe())


# For revives the scenario is very similar, here one can say that the mean is greater just due to top 10% lives for a larger time.
# 
# The conclusion is that teamwork is not really related with a higher final position.
