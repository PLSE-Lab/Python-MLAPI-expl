#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# Cancelled, behind closed doors or in entirely different locations, football is changing for the foreseeable future.
# 
# With some teams positioning themselves against a restart without fans & home games, what can a historical dataset of matches tell us about why these teams worry about losing home advantage?
# 
# Kaggle provides us with a dataset of top 5 league matches from 95/96 - 19/20. This piece is an introduction to the dataset and how to approach measuring home advantage.
# 
# We'll tidy it up and make some new datapoints, before looking at how many points a team should expect when they play at home. Finally, we will look to see if this has changed over time and, if so, whether this is in favour of the stronger or weaker teams in the Premier League in particular.

# In[ ]:


#Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import data and check out the structure
data = pd.read_csv('/kaggle/input/big-five-european-soccer-leagues/BIG FIVE 1995-2019.csv')
data.head()


# ## Tidying & Creating Features
# 
# 
# So we have a match-by-match dataset, covering who played, when, and how many goals/points for each team.
# 
# The dataset has a few things that we can improve on:
# 
# * The date column needs cleaning - break out day, month, year and gameweek
# * We can calculate 2nd half goals
# * Goal difference (GGD) will always be positive, we should have a better one that allows for negatives for away teams
# * And let's get that goal difference by half, too

# In[ ]:


#Create 2nd half goals columns by subtracting HT from FT
data['H2 Team 1'] = data['FT Team 1'] - data['HT Team 1']
data['H2 Team 2'] = data['FT Team 2'] - data['HT Team 2']

#Rename HT to represent 1/2 halves
data.rename(columns={'HT Team 1':'H1 Team 1','HT Team 2':"H1 Team 2"}, inplace=True)

#Goal difference is given, but is positive for either team that wins - let's create a scale where a negative shows an away win
data['FT OGD'] = data['FT Team 1'] - data['FT Team 2']
data['H1 OGD'] = data['H1 Team 1'] - data['H1 Team 2']
data['H2 OGD'] = data['FT OGD'] - data['H1 OGD']

#Rename year to season, to free up year for calendar year
data.rename(columns={'Year':'Season'}, inplace=True)

#Split the date column into the 5 parts of information
data[['Day of Week', 'Day', 'Month', 'Year', 'Gameweek']] = data['Date'].str.split(' ',expand=True)
data['Day of Week'] = data['Day of Week'].str.strip('()')
data['Gameweek'] = data['Gameweek'].str.strip('()')

#Check out the new first row of data
data.iloc[0]


# That looks a lot more useful, now we have a few extra datapoints to compare for home vs away performances.
# 
# This dataset is match-by-match. To compare over time, one approach would be to group a team's home and away performances over a season, then compare from there.
# 
# The code below will create two groups of data, HomePoints & AwayPoints. These will come from grouping the match-level rows by home and away performances, calculating per game figures and merging them together.
# 
# Side note, this could be a lot tidier. My apologies for the repeated code - keep it [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) when you can.

# In[ ]:


#Create HomePoints & AwayPoints - datasets that group matches into teams home and away totals for select colummns
HomePoints = data.groupby(['Season','Country','Team 1']).sum().reset_index()[['Season','Country','Team 1','Team 1 (pts)','FT Team 1', 'H1 Team 1', 'H2 Team 1', 
                                                                              'FT Team 2', 'H1 Team 2', 'H2 Team 2']]
AwayPoints = data.groupby(['Season','Country','Team 2']).sum().reset_index()[['Season','Country','Team 2','Team 2 (pts)','FT Team 2', 'H1 Team 2', 'H2 Team 2', 
                                                                              'FT Team 1', 'H1 Team 1', 'H2 Team 1']]

#Create a column that counts the times a team has played
HomeGames = data.groupby(['Season','Country','Team 1']).count()['Team 1 (pts)']
AwayGames = data.groupby(['Season','Country','Team 2']).count()['Team 2 (pts)']

#Add the matches played count to the HomePoints/AwayPoints dataset
HomePoints = HomePoints.reset_index(drop=True)
HomePoints['Matches'] = HomeGames.reset_index()['Team 1 (pts)']
AwayPoints = AwayPoints.reset_index(drop=True)
AwayPoints['Matches'] = AwayGames.reset_index()['Team 2 (pts)']

#Give the dataset columns some better names
HomePoints.rename(columns={'Team 1':'Team', 'Team 1 (pts)':'Points','FT Team 1' : 'GF', 'H1 Team 1' : 'H1 GF', 'H2 Team 1': 'H2 GF',
                           'FT Team 2': 'GA', 'H1 Team 2': 'H1 GA', 'H2 Team 2': 'H2 GA'},inplace=True)
AwayPoints.rename(columns={'Team 2':'Team', 'Team 2 (pts)':'Points','FT Team 2' : 'GF', 'H1 Team 2' : 'H1 GF', 'H2 Team 2': 'H2 GF',
                           'FT Team 1': 'GA', 'H1 Team 1': 'H1 GA', 'H2 Team 1': 'H2 GA'},inplace=True)

#Add a new column to each to tell us that it was home or away before we merge them
HomePoints['Location'] = 'Home'
AwayPoints['Location'] = 'Away'

#Stick AwayPoints to the end of HomePoints and save it to MergePoints
MergePoints = HomePoints.append(AwayPoints)

#Calculate per game metrics and put them in new columns
for col in MergePoints.columns[3:10]:
    colname = str(col) + ' PG'
    MergePoints[colname] = round(MergePoints[col]/MergePoints['Matches'],1)
    
#Check it out
MergePoints.head()


# Looks great! Team by season, split by home/away - should be loads to get stuck into here.
# 
# We might also want to look on a league by season level, so let's create that quickly...

# In[ ]:


MergePointsSeason = MergePoints.groupby(['Season','Country','Location']).mean().reset_index()
MergePointsSeason.head()


# ## Exploratory Analysis
# 
# We now have three datasets:
# 
# 1. Data - match-by-match level
# 2. MergePoints - team-season level, split by H and A games
# 3. MergePointsSeason - league-level, split by H and A games
# 
# With this, we can now look to explore the difference in goals and points between home and away games. Firstly, let's take a quick look at the entire dataset altogether, before splitting out into leagues.

# In[ ]:


data.describe()


# At the match level, considering all 5 leagues together, on average the home team (FT Team 1 vs FT Team 2) wins the match by 1.54 to 1.11 - and the average home team should expect 1.7 points, while the typical away team should expect 1.1 points.
# 
# Let's check out the games with a goal difference of 9, too.

# In[ ]:


data[data['GGD']==9]


# No surprise to see PSG in twice!
# 
# Let's now visualise the frequency of wins, draws and losses. Using the match-level data again, plot the distribution of the home team's points:

# In[ ]:


sns.distplot( a=data['Team 1 (pts)'], hist=True, kde=False, rug=False)
plt.xlim(0,3)
plt.show()


# And do the same for the goal difference from the perspective of the home team. In this plot, +1 represents a 1-goal win for the home team and -2 represents a 2-goal loss

# In[ ]:


sns.distplot( a=data['FT OGD'], bins=range(-9,9), hist=True, kde=False, rug=False, norm_hist=True)
plt.xlim(-9,9)
plt.xlabel('Home Team Goal Difference in a Single Game')
plt.show()


# 0 - a draw - is the most common goal difference, over a quarter of the time. Followed closely by a 1-goal win. After this, there is a drop to around 15% of matches ending with either a 1-goal away win or 2-goal home win.
# 
# So we know where we are for all matches in the dataset from 1995 onwards. But football has changed a fair bit since then, maybe home vs away performance has also shifted? Let's take a look:

# In[ ]:


plt.plot(data.groupby('Season')['Team 1 (pts)'].mean())
plt.plot(data.groupby('Season')['Team 2 (pts)'].mean())
plt.title('Home vs Away Points per Game, 1995/96-2019/20')
plt.show()


# The trend for home points per game is dropping, meaning that away points have to rise. The changes look a bit less jagged on the away average, likely because the points jump from a loss to a draw is half that of a draw to a win.
# 
# With the Premier League likely moving into neutral venues, and other competitions at least going behind closed doors, let's now check out this trend split by league:

# In[ ]:


g = sns.FacetGrid(MergePoints.groupby(['Season','Country','Location']).mean().reset_index(), row="Country", col="Location")
g = g.map(plt.plot, "Points PG")


# Each league to significantly different levels, but the reduction in home advantage looks to be present across all leagues. Most dramatically in Italy, and less so in England.
# 
# The next plot will focus on England, taking points per game again, and also goals for per game. This way, we might be able to see if the strengthening away performances are due to more goals, or better defences.

# In[ ]:


#In EPL, How have points per game and goals for per game changed over time?
g = sns.FacetGrid(MergePointsSeason[MergePointsSeason['Country'] == 'ENG'], col="Location")
g = g.map(plt.plot, 'Points PG', color='blue')
g = g.map(plt.plot, 'GF PG', color = 'orange')


# Away goals look to be the main difference over time here. First half, or second half?

# In[ ]:


#In EPL, goals per half
g = sns.FacetGrid(MergePointsSeason[MergePointsSeason['Country'] == 'ENG'], col="Location")
g = g.map(plt.plot, 'H1 GF PG', color='pink')
g = g.map(plt.plot, 'H2 GF PG', color = 'green')


# So we know that away points per game has increased, reducing home points per game. And we can see that this is on account of a steady increase in away goals scored, increasing in both halves, but moreso in the second.
# 
# So while home advantage still exists (.5-.6 points per game's worth), it is weaker than it has been. And due to away goals being scored more often, it would be worth checking whether or not this is due to weaker teams getting results, or stronger teams pushing their advantage further.
# 
# One way to do this is to examine the spread of away points per game away, to see if teams are getting closer or futher away from eachother.
# 
# Let's visualise this with a boxplot, showing the spread of points per game for away teams only in the Premier League:

# In[ ]:


sns.set(rc={'figure.figsize':(14,7)})
ax = sns.boxplot(x="Season", y="Points PG", hue="Location", data=MergePoints[(MergePoints['Country'] == 'ENG') & (MergePoints['Location'] == 'Away')])
plt.title('Distribution of EPL Away Points per Game: 1995-2019')
plt.show()


# Key takeaways from this chart:
# 
# 1. The spread is increasing over time
# 2. The spread is due to more teams reporting higher PPG away
# 3. The lower end of the league is much more consistent than the top end
# 
# Altogether, the reduction in home advantage seems to be on account of a stronger showing from the top teams, rather than more upsets.
# 
# We can plot the bottom and top quantiles of the home and away performance, to see the trend for both:

# In[ ]:


plt.figure()


plt.subplot(121)
plt.plot(MergePoints[(MergePoints['Country'] == 'ENG') & (MergePoints['Location'] == 'Home')].groupby(['Season'])['Points PG'].quantile(.75),
         label = 'Upper Quartile Home PPG', color='firebrick')
plt.plot(MergePoints[(MergePoints['Country'] == 'ENG') & (MergePoints['Location'] == 'Home')].groupby(['Season'])['Points PG'].quantile(.25),
         color='lightcoral', label = 'Lower Quartile Home PPG')
plt.legend()
plt.ylabel('Points Per Game', fontsize=12)
plt.ylim(0.5, 2.5)


plt.subplot(122)

plt.plot(MergePoints[(MergePoints['Country'] == 'ENG') & (MergePoints['Location'] == 'Away')].groupby(['Season'])['Points PG'].quantile(.75), 
         label='Upper Quartile Away PPG', color='royalblue')
plt.plot(MergePoints[(MergePoints['Country'] == 'ENG') & (MergePoints['Location'] == 'Away')].groupby(['Season'])['Points PG'].quantile(.25),
         label='Lower Quartile Away PPG', color='dodgerblue')
plt.legend()

plt.ylabel('Points Per Game', fontsize=12)
plt.suptitle('Changes in home & away EPL performance, by top & bottom groups of teams', fontsize=16)
plt.ylim(0.5, 2.5)

plt.show()


# So now we can see quite clearly that the reduction in home advantage only seems to take place among the bottom teams. Top teams have maintained their dominance at home, while also gaining away points. As was suspected before, it is indeed a widening of the top and bottom teams that has caused the reduction in league-wide home points per game.
# 
# ## Conclusion
# 
# This piece aimed to introduce one approach to this dataset. We started by creating new variables to show half-by-half and overall home/away performance on a match, team and season level. We then explored how H/A evolved over time and tried to unpack why these changes happened.
# 
# We found that home advantage overall has dropped over time in each of the top 5 leagues. In the Premier League, this is driven in part by more goals being scored by away teams. Additionally, when we split out H/A form by the level of the team, the drop in home performance is only seen among the poorer teams. The better teams in the league are maintaining their home level AND improving their away form, demonstrating the widening gap in Premier League football.
# 
# There are loads of other questions to answer both with this dataset and by enhancing it with more data:
# 
# * Dives into other leagues
# * Is home advantage affected by time in season? Does it mean more when survival or European places are on the line?
# * Individual team profiles
# * Why home advantage is even a thing
# 
# Soon, I imagine we might also be more knowledgable on the effects of empty and neutrally held matches...
# 
# I look forward to seeing where people can enhance this work - please show us at [FC Python](https://twitter.com/FC_Python) and we will be happy to share!
