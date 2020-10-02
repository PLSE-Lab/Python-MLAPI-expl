#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import trueskill # trueskill ranking

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # TrueSkill Ranking on English Premier League Dataset

# In this notebook, I applied [TrueSkill](https://trueskill.org/) rating system to calculate ranks of English Premiere League teams. The data is sourced from the [Kaggle datasets](https://www.kaggle.com/irkaal/english-premier-league-results#EPL.csv). 
# My objevtive was to learn and implement TrueSkill ranking using real data.
# 
# TrueSkill is a rating system which was developed by Microsoft to rank and match players. It is based on ELO ranking system, but gives more flexibility, like multiplayer games ranking.
# 
# Ranking starts with 0 and goes up to 50 points.The TrueSkill rank is calculated based on the Score = mu - 3* sigma and could be considered as conservative estimation of the players' skills, as system is 99% confident that players' skills are higher than calculated Scores. 
# 
# Starting scores by default: mu = 25, sigma=8.33. For this exercise I will use score = (mu - 2*sigma) instead, so starting scores will be 8.33, and not 0.

# ### Read in data.

# In[ ]:


df = pd.read_csv('../input/english-premier-league-results/EPL.csv')
df.head().append(df.tail())


# In[ ]:


df.columns


# ### Leave columns required for TrueSkill ranking implementation   

# In[ ]:


df = df[['Date', 'HomeTeam', 'AwayTeam','FTR']].copy()
df.head()


# In[ ]:


set(df['AwayTeam'].unique())==set(df['HomeTeam'].unique())


# ### 'FTR' - Full Time Result (H=Home Win, D=Draw, A=Away Win)
# #### For TrueSkill Win=0,Draw=0,Loose=1. Implement this accordingly.  

# In[ ]:


df['Result_HomeTeam'] = df['FTR']
df['Result_AwayTeam'] = df['FTR']

df.replace({'Result_HomeTeam': {'H':0,'D':0,'A':1}},inplace=True)
df.replace({'Result_AwayTeam': {'H':1,'D':0,'A':0}},inplace=True)
df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


teams = df['HomeTeam'].unique().tolist()
ts = trueskill.TrueSkill()

ranking = []

# Create Default ratings(mu=25, sigma=8.33)
for team in teams:
       ranking.append(ts.create_rating())

# create dictionary with all teams and initial ratings
all_ranks_dict = dict(zip(teams,ranking))
all_ranks_dict


# In[ ]:


home_team_rank = df['Result_HomeTeam'].values
away_team_rank = df['Result_AwayTeam'].values
ts_ranks = np.stack((home_team_rank,away_team_rank), axis=-1) # create array of arrays with results

home_team = df['HomeTeam'].values
away_team = df['AwayTeam'].values
match_array = np.stack((home_team,away_team), axis=-1) # create array of arrays with all matches


# In[ ]:


def rating(mu,sigma):
    """
    mu and sigma from TrueSkill: 
    mu = ts_rating.mu;
    sigma = ts_rating.sigma
    
    Function returns trueskill rating value ('real score' is with 97.1% confidence not below that value)
    """
    return mu-2*sigma


# In[ ]:


# Create lists: with current ranks(before the game) and new ranks(after the game)
curr_ranks_list=[]
new_ranks_list=[]
for i in range(len(match_array)):
    
    # current ranks:
    home_team_rank = all_ranks_dict[match_array[i][0]]
    away_team_rank = all_ranks_dict[match_array[i][1]]

    curr_ranks_list.append([rating(home_team_rank.mu,
                                   home_team_rank.sigma),
                            rating(away_team_rank.mu,
                                   away_team_rank.sigma)])
    
    # new ranks:
    new_ranks = ts.rate([(home_team_rank,),
                         (away_team_rank,)],
                        ranks = ts_ranks[i])
    
    new_home_team_rank = new_ranks[0][0]
    new_away_team_rank = new_ranks[1][0]
    
    new_ranks_list.append([rating(new_home_team_rank.mu,
                                  new_home_team_rank.sigma),
                           rating(new_away_team_rank.mu,
                                  new_away_team_rank.sigma)])        
    
    # update dictionary with changed/new ranks:
    all_ranks_dict[match_array[i][0]] = new_home_team_rank
    all_ranks_dict[match_array[i][1]] = new_away_team_rank


# In[ ]:


### Combine results in one dataframe 
df = pd.concat([df,
                pd.DataFrame(curr_ranks_list,columns=['Rank_HT_Before', 'Rank_AT_Before']),
                pd.DataFrame(new_ranks_list, columns= ['Rank_HT_After', 'Rank_AT_After'])],
               axis=1)

df.drop(columns=['Result_HomeTeam','Result_AwayTeam'],inplace=True)
df.head().append(df.tail(10))


# In[ ]:


### Get the latest TrueSkill Ranks.
latest = pd.DataFrame(all_ranks_dict).transpose()
latest.columns = ['mu','sigma']
latest['rank']=rating(latest['mu'],latest['sigma'])
latest.sort_values(by='rank', ascending=False)

