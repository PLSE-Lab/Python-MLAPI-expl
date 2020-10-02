#!/usr/bin/env python
# coding: utf-8

# This is an analysis of IPL 1-9 with regards to players as a resource.

# In[10]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[11]:


deliveries = pd.read_csv('../input/deliveries.csv')
matches = pd.read_csv('../input/matches.csv')


# In[ ]:


matches.columns


# In[ ]:


deliveries.columns


# In[ ]:


def add_batsmen(roster,sample):
    batsmen = sample.groupby('batsman')
    for batsman in batsmen.groups:
        teams = batsmen.get_group(batsman).groupby('batting_team')
        for team in teams.groups:
            roster=roster.append(pd.DataFrame({'type':'batsmen','player':batsman,'team':team, 'match_id':teams.get_group(team)['match_id'].unique()}))    
    return roster

def add_bowlers(roster,sample):
    bowlers = sample.groupby('bowler')
    for bowler in bowlers.groups:
        teams = bowlers.get_group(bowler).groupby('bowling_team')
        for team in teams.groups:
            roster=roster.append(pd.DataFrame({'type':'bowler','player':bowler,'team':team, 'match_id':teams.get_group(team)['match_id'].unique()}))
    return roster


# In[ ]:


def player_roster(size):
    sample = deliveries.head(size)
    roster = pd.DataFrame()
    roster = add_batsmen(roster,sample)
    roster = add_bowlers(roster,sample)
    return roster
    


# In[ ]:


def find_all_rounder(roster):
    print(roster.groupby('player')['type'].count()>1)


# In[ ]:


def yoy_toss_cities(sample):
    
    results = pd.DataFrame()
    seasons = sample.groupby('season')
    for season in seasons.groups:
        print(city_toss(seasons.get_group(season)))
    return results 


# In[ ]:


def important_toss_label(sample):
    results = pd.DataFrame()
    sample.loc[:,'toss_importance']=sample['toss_winner']==sample['winner']
    return sample


# In[ ]:


def city_toss(sample):
    results = pd.DataFrame()
    sample = important_toss_label(sample)
    cities = sample.groupby('city')
    for city in cities.groups:
        results = results.append(pd.DataFrame({'city':city,'TIP':np.count_nonzero(cities.get_group(city)['toss_importance'])/cities.get_group(city)['toss_importance'].count()}))


# In[ ]:


def close_failed_chases():
    seasons = matches[matches['win_by_runs']!=0].groupby('season')
    results = pd.Series()
    for season in seasons.groups:
        print(season, seasons.get_group(season)['win_by_runs'].mean())
#plt.scatter(results[0],results[1])
#plt.show()


# In[ ]:


deliveries.head(123)['total_runs'].sum()


# In[ ]:


def match_scores(sample,matches):
    matches_played = sample.groupby('match_id')
    for match in matches_played.groups:
        first_total = 0
        second_total = 0
        first_wickets = 0
        second_wickets = 0
        innings = matches_played.get_group(match).groupby('inning')
        first = innings.get_group(1)
        first_total=first['total_runs'].sum()
        first_wickets = first[first['player_dismissed']!=""]['player_dismissed'].count()
        if len(innings.groups)==2:    
            second = innings.get_group(2)
            second_total=second['total_runs'].sum()
            second_wickets = second[second['player_dismissed']!=""]['player_dismissed'].count()
        matches.loc[matches['id']==match,'target'] = first_total 
        matches.loc[matches['id']==match,'chase'] = second_total
        print(first_total,first_wickets,second_total,second_wickets)
    return matches


# In[ ]:


augmented = match_scores(deliveries.head(900),matches)


# In[ ]:


def find_extra_balls(sample):
    return sample[sample['noball_runs']!=0]['noball_runs'].count()+sample[sample['wide_runs']!=0]['wide_runs'].count()

def find_scoring_shots_with(sample,runs):
    scoring_shot = sample[sample['batsman_runs']==runs]
    return scoring_shot['over']*6+scoring_shot['ball']
    
    #return scoring_shot['over']*6+scoring_shot['ball']

def find_scoring_shots(sample):
    return sample[sample['batsman_runs']!=0]['batsman_runs'].count()
    
def find_dots(sample):
    return sample[sample['total_runs']==0]['total_runs'].count()

def find_boundaries_runs(sample):
    return sample[(sample['batsman_runs']==4) | (sample['batsman_runs']==6)]['batsman_runs'].sum()

def find_byes_legbyes(sample):
    return sample[(sample['bye_runs']!=0) | (sample['legbye_runs']!=0)]['extra_runs'].count()
    


# In[ ]:


def process_inning(innings_details, inning):
    batsman_performance = pd.DataFrame()
    batsmen = match_details.groupby('batsman')
    for batsman in batsmen.groups:
        batsman_stats = batsmen.get_group(batsman)
        runs = batsman_stats['batsman_runs'].sum()
        balls = batsman_stats['batsman_runs'].count()-find_extra_balls(batsman_stats)
        dots = find_dots(batsman_stats)
        scoring_shots=find_scoring_shots(batsman_stats)
        byes_legbyes = find_byes_legbyes(batsman_stats)
        boundaries_score = find_boundaries_runs(batsman_stats)
        fours = find_scoring_shots_with(batsman_stats,4)
        sixes = find_scoring_shots_with(batsman_stats,6)
        batsman_performance = batsman_performance.append(pd.DataFrame({'match':[match],'batsman':[batsman], 'runs': [runs], 'balls':[balls], 'dots':[dots], 'scoring_shots':[scoring_shots], 'boundaries_score': [boundaries_score], 'fours': [fours], 'sixes':[sixes],'innings':innings}))
        #print(batsman,batsman_stats[batsman_stats['noball_runs']!=0]['noball_runs'].count()+batsman_stats[batsman_stats['wide_runs']!=0]['wide_runs'].count())
        #pd.DataFrame({'batsman':batsman, 'runs':runs, 'balls':})
    return batsman_performance


# In[ ]:


matches_played = deliveries.head(1000).groupby('match_id')
batsman_performance = pd.DataFrame()
for match in matches_played.groups:
    innings = matches_played.get_group(match).groupby('inning')
    batsman_performance.append(process_inning(innings.get_group(inning),1))
    process_inning(innings.get_group(inning),2)
    


# In[ ]:


batsman_performance.reset_index(drop=True)

