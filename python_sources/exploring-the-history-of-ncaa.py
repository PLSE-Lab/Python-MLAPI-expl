#!/usr/bin/env python
# coding: utf-8

# This kernel is a work in progress
# 
# The goal of this kernel is to explore the data available for this competition and have some fun in exploring how the NCAA has changed (or not changed) since 1985. Since we have detailed results since 2003, we can also try to find how winning and losing teams differ from one another.

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Teams, games, and victories
# 
# Since 1985, the Division 1 has seen a growth in the number of competing teams. This can be observed from the file called `Teams.csv`, a file where we have the first and last year that the team competed in Division 1 (the teams that joined before 1985 are recorded as if they joined in 1985)

# In[ ]:


teams = pd.read_csv('../input/datafiles/Teams.csv')
teams.head()


# In[ ]:


teams.FirstD1Season.hist(bins=20, alpha=0.7, label='First Season', figsize=(8,5))
teams.LastD1Season.hist(bins=20, alpha=0.7, label='Last Season')
plt.title('Number of teams that joined or left the Division 1', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(False)
plt.legend()


# As we see, almost every team is there since the beginning and never left. We can see this better if we plot the number of teams every year

# In[ ]:


yr_count = pd.DataFrame({'year': np.arange(1985, 2020)})

for year in yr_count.year:
    teams['is_in'] = 0
    teams.loc[(teams.FirstD1Season <= year) & (teams.LastD1Season >= year), 'is_in'] = 1
    tot_teams = teams.is_in.sum()
    yr_count.loc[yr_count.year == year, 'n_teams'] = tot_teams
    
yr_count = yr_count.set_index('year')
yr_count.n_teams.plot(figsize=(12,6))
plt.title('Number of teams in Division 1', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('N. Teams', fontsize=12)


# ## Regular Season results, 1985-2018
# 
# We also have some details about every game of the regular season in `RegularSeasonCompactResults.csv`

# In[ ]:


reg_season = pd.read_csv('../input/datafiles/RegularSeasonCompactResults.csv')
reg_season.head()


# In[ ]:


reg_season['point_diff'] = reg_season.WScore - reg_season.LScore
reg_season.point_diff.hist(bins=30, figsize=(10,5))
plt.title('Point difference in regular season', fontsize=15)
plt.xlabel('Point difference', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(False)


# We see that there have been a few blowouts (notably, a game in 1996 that ended with 91 points of difference). We can have a more detailed view by creating a small dataframe to summarize the results of each year

# In[ ]:


summaries = reg_season[['Season', 
    'WScore', 
    'LScore', 
    'NumOT', 
    'point_diff']].groupby('Season').agg(['min', 'max', 'mean', 'median'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries.sample(10)


# In[ ]:


fig, ax= plt.subplots(2,2, figsize=(15, 12))

wscores = [col for col in summaries.columns if 'WScore' in col]
lscores = [col for col in summaries.columns if 'LScore' in col]
point_diffs = [col for col in summaries.columns if 'point_diff' in col]

summaries[wscores].plot(ax=ax[0][0], title='Scores of the winning teams', ylim=(15, 190))
ax[0][0].legend(labels=['Min', 'Max', 'Mean', 'Median'])
summaries[lscores].plot(ax=ax[0][1], title='Scores of the losing teams', ylim=(15, 190))
ax[0][1].legend(labels=['Min', 'Max', 'Mean', 'Median'])
summaries[point_diffs].plot(ax=ax[1][0], title='Point differences')
ax[1][0].legend(labels=['Min', 'Max', 'Mean', 'Median'])
summaries[['NumOT_mean']].plot(ax=ax[1][1], title='Average number of OT')
ax[1][1].legend(labels=['Mean'])


# We see that every year there is always a blowout but, in general, the average number of points stayed the same since 1985. Interesting to notice how the mean number of overtimes had a sudden growth in the year 2000. It can be either a problem with the data or a change in the regulations (but I couldn't find any reference to it in my lazy google search).
# 
# Another visible trend emerges if we summarize considering where the game was played.

# In[ ]:


summaries = reg_season[['Season', 'WLoc',
    'WScore', 
    'LScore', 
    'NumOT', 
    'point_diff']].groupby(['Season', 'WLoc']).agg(['min', 'max', 'mean', 'median', 'count'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries.sample(5)


# In[ ]:


fig, ax= plt.subplots(3,2, figsize=(15, 18))

wscores = [col for col in summaries.columns if 'WScore' in col]
lscores = [col for col in summaries.columns if 'LScore' in col]
point_diffs = [col for col in summaries.columns if 'point_diff' in col]

summaries[['WScore_mean']].unstack().plot(ax=ax[0][0], title='Avg. scores of the winning teams', ylim=(60,87))
ax[0][0].legend(labels=['Away', 'Home', 'Neutral'])
summaries[['LScore_mean']].unstack().plot(ax=ax[0][1], title='Avg. scores of the losing teams', ylim=(60,87))
ax[0][1].legend(labels=['Away', 'Home', 'Neutral'])
summaries[['point_diff_mean']].unstack().plot(ax=ax[1][0], title='Avg. point differences')
ax[1][0].legend(labels=['Away', 'Home', 'Neutral'])
summaries[['WScore_count']].unstack().plot(ax=ax[1][1], title='Number of wins by location')
ax[1][1].legend(labels=['Away', 'Home', 'Neutral'])
summaries[['NumOT_mean']].unstack().plot(ax=ax[2][0], title='Average number of OT')
ax[2][0].legend(labels=['Away', 'Home', 'Neutral'])
summaries[['NumOT_max']].unstack().plot(ax=ax[2][1], title='Maximum number of OT')
ax[2][1].legend(labels=['Away', 'Home', 'Neutral'])


# We see some interesting (although not so surprising) patterns. ***Note***: remember that the legend refers to whether or not the winning team was at home or not.
# 
# * If the team is at home, it scores on average more points (this can be seen in the first 2 graphs, meaning that it does so regardless of the outcome of the game).
# * There is no difference between a team away or on a neutral location if we just look at the point scored
# * The home team tends to win more often and neutral locations are rarer.
# * On average, a team at home wins with more points difference
# * The away team goes to OT more often in order to get its win (which is coherent with the previous point)
# 
# I was always wondering if during the season the probability of a blow out was changing in any way but the next graph tells me that I have to put more effort than that

# In[ ]:


plt.figure(figsize=(12,10))
sns.scatterplot(reg_season.DayNum, reg_season.point_diff)


# For the more recent seasons (2003 - 2018), we have more detailed results in the file `RegularSeasonDetailedResults.csv`. We extend this file by computing some extra statistics and other things that might be useful.

# In[ ]:


def process_details(df):
    data = df.copy()
    stats = [col for col in data.columns if 'W' in col and 'ID' not in col and 'Loc' not in col]

    for col in stats:
        name = col[1:]
        data[name+'_diff'] = data[col] - data['L'+name]
        data[name+'_binary'] = (data[name+'_diff'] > 0).astype(int)
        
    for prefix in ['W', 'L']:
        data[prefix+'FG_perc'] = data[prefix+'FGM'] / data[prefix+'FGA']
        data[prefix+'FGM2'] = data[prefix+'FGM'] - data[prefix+'FGM3']
        data[prefix+'FGA2'] = data[prefix+'FGA'] - data[prefix+'FGA3']
        data[prefix+'FG2_perc'] = data[prefix+'FGM2'] / data[prefix+'FGA2']
        data[prefix+'FG3_perc'] = data[prefix+'FGM3'] / data[prefix+'FGA3']
        data[prefix+'FT_perc'] = data[prefix+'FTM'] / data[prefix+'FTA']
        data[prefix+'Tot_Reb'] = data[prefix+'OR'] + data[prefix+'DR']
        data[prefix+'FGM_no_ast'] = data[prefix+'FGM'] - data[prefix+'Ast']
        data[prefix+'FGM_no_ast_perc'] = data[prefix+'FGM_no_ast'] / data[prefix+'FGM']
        
    data['Game_Rebounds'] = data['WTot_Reb'] + data['LTot_Reb']
    data['WReb_perc'] = data['WTot_Reb'] / data['Game_Rebounds']
    data['LReb_perc'] = data['LTot_Reb'] / data['Game_Rebounds']
    
    return data


# In[ ]:


reg_season = pd.read_csv('../input/datafiles/RegularSeasonDetailedResults.csv')

stats = [col for col in reg_season.columns if 'W' in col and 'ID' not in col and 'Loc' not in col]

reg_season = process_details(reg_season)

reg_season.head()


# In[ ]:


not_sum = ['WTeamID', 'DayNum', 'LTeamID']
to_sum = [col for col in reg_season.columns if col not in not_sum]

summaries = reg_season[to_sum].groupby(['Season', 'WLoc']).agg(['min', 'max', 'mean', 'median', 'count'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]
summaries.sample(5)


# In[ ]:


fig, ax= plt.subplots(7,2, figsize=(15, 6*7))

i, j = 0, 0

for col in stats:
    name = col[1:]
    summaries[[c for c in summaries.columns if name+'_diff_mean' in c]].unstack().plot(title='Difference in mean '+name,ax=ax[i][j])
    ax[i][j].legend(labels=['Away', 'Home', 'Neutral'])
    if j == 0: j = 1
    else:
        j = 0
        i += 1


# Let's put some order to these findings (please note we are looking at average values)
# 
# * Not surprisingly, who scores more FG wins and, as before, we see that the spread between the winning and the losing teams is bigger if the winning team plays at home
# * More surprisingly, the winning team is attempting fewer field goals than the losing one and we observe again that the spread gets smaller if the winning team plays at home. However, since 2016 this is not true anymore
# * There is an increasing trend in the number of 3 pointers made and this is more evident when the losing team is playing at home
# * Opposite trend in the number of free throws, this could indicate that, generally, the teams are adapting their defense to the new regulations that allow fewer contacts
# * The winning team grabs more rebounds but for the offensive rebounds we see that at home the winning team is showing more commitment (or simply, if you play at home you better get those rebounds)
# * Not surprisingly, the winning team values their possessions more (thus fewer turnovers) and commits fewer fouls (at home in particular)
# * The winning team puts more effort in their defense, but away they just needed to match the effort of their opponents to win.
# 
# Before moving to the playoffs, let's have a look at some of the stats we created

# In[ ]:


fig, ax= plt.subplots(6,2, figsize=(15, 6*6))

i = 0

for col in [c for c in summaries.columns if '_perc_mean' in c and c.startswith('W')]:
    name = col.split('_perc_')[0][1:]
    summaries[col].unstack().plot(title='Mean percenteage of '+name+', Winners',ax=ax[i][0])
    summaries['L'+name+'_perc_mean'].unstack().plot(title='Mean percenteage of '+name+', Losers',ax=ax[i][1])
    ax[i][0].legend(labels=['Away', 'Home', 'Neutral'])
    ax[i][1].legend(labels=['Away', 'Home', 'Neutral'])
    i += 1


# We thus see a few new patterns:
# 
# * FG and FT accuracies are increasing a lot in the past 2-3 years
# * The winning team is, of course, generally more efficient (more accurate, more rebounds, etc.)
# * The losing team scores on average a larger proportions of their points **without an assist**, which can be an indication that they do not share the ball enough.

# In[ ]:





# ## March madness
# 
# Now we can turn our attention towards the playoff and see if we can spot any difference. It is now time to have a look at `NCAATourneyCompactResults.csv` 

# In[ ]:


playoff = pd.read_csv('../input/datafiles/NCAATourneyCompactResults.csv')
playoff.head()


# In[ ]:


playoff['point_diff'] = playoff.WScore - playoff.LScore
playoff.point_diff.hist(bins=30, figsize=(10,5))
plt.title('Point difference in the playoffs', fontsize=15)
plt.xlabel('Point difference', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(False)


# In[ ]:


summaries = playoff[['Season', 
    'WScore', 
    'LScore', 
    'NumOT', 
    'point_diff']].groupby('Season').agg(['min', 'max', 'mean', 'median'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]

fig, ax= plt.subplots(2,2, figsize=(15, 12))

wscores = [col for col in summaries.columns if 'WScore' in col]
lscores = [col for col in summaries.columns if 'LScore' in col]
point_diffs = [col for col in summaries.columns if 'point_diff' in col]

summaries[wscores].plot(ax=ax[0][0], title='Scores of the winning teams', ylim=(25, 160))
ax[0][0].legend(labels=['Min', 'Max', 'Mean', 'Median'])
summaries[lscores].plot(ax=ax[0][1], title='Scores of the losing teams', ylim=(25, 160))
ax[0][1].legend(labels=['Min', 'Max', 'Mean', 'Median'])
summaries[point_diffs].plot(ax=ax[1][0], title='Point differences')
ax[1][0].legend(labels=['Min', 'Max', 'Mean', 'Median'])
summaries[['NumOT_mean']].plot(ax=ax[1][1], title='Average number of OT')
ax[1][1].legend(labels=['Mean'])


# We thus see how during March Madness teams put more effort in their defense, the blowouts are less extreme, and (on average) the winning and losing teams are scoring a more similar amount of points with respect to the regular season results. As before, there is no visible pattern over time but this time we do not observe that weird increase in the OT rate.
# 
# All these results should be cross-checked with the locations of the games because, from this table, we get them all happening at a neutral location.

# In[ ]:


playoff = pd.read_csv('../input/datafiles/NCAATourneyDetailedResults.csv')

stats = [col for col in playoff.columns if 'W' in col and 'ID' not in col and 'Loc' not in col]

playoff= process_details(playoff)

not_sum = ['WTeamID', 'DayNum', 'LTeamID']
to_sum = [col for col in reg_season.columns if col not in not_sum]

summaries = playoff[to_sum].groupby(['Season']).agg(['min', 'max', 'mean', 'median', 'count'])

summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]

fig, ax= plt.subplots(7,2, figsize=(15, 6*7))

i, j = 0, 0

for col in stats:
    name = col[1:]
    summaries[[c for c in summaries.columns if name+'_diff_mean' in c]].plot(title='Difference in mean '+name,ax=ax[i][j])
    if j == 0: j = 1
    else:
        j = 0
        i += 1


# This time we don't observe the trends we observed for the average number of 3 pointers and free throws in the regular season.

# In[ ]:


fig, ax= plt.subplots(6,2, figsize=(15, 6*6))

i = 0

for col in [c for c in summaries.columns if '_perc_mean' in c and c.startswith('W')]:
    name = col.split('_perc_')[0][1:]
    summaries[col].plot(title='Mean percenteage of '+name+', Winners',ax=ax[i][0])
    summaries['L'+name+'_perc_mean'].plot(title='Mean percenteage of '+name+', Losers',ax=ax[i][1])
    i += 1


# We see generally the same behavior we observed for the regular season, with some exceptions
# 
# * The accuracy in 3 pointers has a slightly negative trend year by year for the winning team
# * The differences between winning and losing teams are less evident (as before)
# * There is a positive trend in the fraction of FG scored without an assist

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## ***To be continued, thank you for reading this far***
