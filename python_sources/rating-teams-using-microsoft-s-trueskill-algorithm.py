#!/usr/bin/env python
# coding: utf-8

# Microsoft has created an algorithm called *TrueSkill* for the purpose of ranking players using its video game platform.  It is basically a probabalistic algorithm based on the *Elo* algorithm used to rank chess players.  It can be used to rank players in any activity where there is some concept of win/lose/draw.  The algorithm itself is somewhat hard to understand but fortunately someone has created a Python package which makes it easy to use without understanding the theory behind it. 

# First we read in the data files we need.

# In[ ]:


import pandas as pd 

results = pd.read_csv('../input/stage2datafiles/RegularSeasonCompactResults.csv')
tourney = pd.read_csv('../input/stage2datafiles/NCAATourneyCompactResults.csv')
seeds = pd.read_csv('../input/stage2datafiles/NCAATourneySeeds.csv')
teams = pd.read_csv('../input/stage2datafiles/Teams.csv')


# Next we define the function which actually uses TrueSkill to rank the teams in a given season.

# In[ ]:


from trueskill import Rating, rate_1vs1
from collections import defaultdict

def get_ratings(season):         
    # start all teams with a default rating
    ratings = defaultdict(Rating)         
    # get data for season
    current_results = results[results['Season'] == season]                                           
    # at the start, all teams are equal which is not realistic so we loop
    # through the season's games several times to get better starting ratings
    for epoch in range(10):                                 
        # loop through the games in order
        for _, row in current_results.sort_values('DayNum').iterrows():                                                    
            wteamid = row['WTeamID']                                                                 
            lteamid = row['LTeamID']    
            # have TrueSkill compute new ratings based on the game result
            ratings[wteamid], ratings[lteamid] = rate_1vs1(ratings[wteamid], ratings[lteamid])       
    # just keep the mean rating
    return {team_id: rating.mu for team_id, rating in ratings.items()}


# The above function is a bit slow, so we'll use multiprocessing to compute ratings for each season in parallel.

# In[ ]:


from multiprocessing import Pool

p = Pool()    
seasons = results['Season'].unique()
ratings = p.map(get_ratings, seasons)                                                                
p.close()                                                                                            
p.join() 

# put ratings into a dict for easy access
ratings = dict(zip(seasons, ratings))

# lets take a look at 2019 rankings
team_names = dict(zip(teams['TeamID'], teams['TeamName']))
ratings_2019 = [(team_names[t], r) for t, r in ratings[2019].items()]
pd.DataFrame(ratings_2019, columns=['TeamID', 'Rating']).sort_values('Rating', ascending=False)


# So now that we have ratings we can try to turn them into probabilities for the 2019 tournament games.  We will do this by using results from past NCAA tournament games with the ratings as features and training a logistic regression model.

# In[ ]:


train = []                                                                                           
target = []              
# create training data with past tournament results
for _, row in tourney.iterrows():                                                                    
    season = row['Season']                                                                           
    wteamid = row['WTeamID']                                                                         
    lteamid = row['LTeamID']                                                                         
    # we add two rows per game so the target is not all '0' or '1'
    # it might be better to randomly choose winner or loser first
    # or always have higher ratings first
    train.append([ratings[season][wteamid], ratings[season][lteamid]])                               
    target.append(1)                                                                                 
    train.append([ratings[season][lteamid], ratings[season][wteamid]])                               
    target.append(0)     
train = pd.DataFrame(train, columns=['Team1', 'Team2'])
target = pd.Series(target, name='Target')
pd.concat((train, target), axis=1)


# OK, now we can train a model to give winning probability given two ratings.

# In[ ]:


from sklearn.linear_model import LogisticRegression                                                  
from sklearn.preprocessing import StandardScaler 

ss = StandardScaler()                                                                                                                                
train = ss.fit_transform(train)                                                            
lr = LogisticRegression()                                                                            
lr.fit(train, target) 
'intercept: {} coefficients: {}'.format(lr.intercept_[0], lr.coef_[0])


# Now that we have our model, we can predict winning probabilities for 2019 tournament games.  First we need to build a test set similar to the training set.

# In[ ]:


# get seeds for 2019 tournament
seeds2019 = seeds['TeamID'][seeds['Season'] == 2019].unique() 
# loop though every possible matchup
predictions = []
for team1 in seeds2019:                                                                                  
    for team2 in seeds2019:                                                                              
        if team1 < team2:
            # we're going to get probabilites for team1 vs team2 and team2 vs team1 and average them
            test_rows = [                                                                            
                [ratings[2019][team1], ratings[2019][team2]],                                        
                [ratings[2019][team2], ratings[2019][team1]],                                        
            ]                                                                                        
            test_rows = ss.transform(test_rows)                                                      
            prob = lr.predict_proba(test_rows)[:, 1]                                                 
            avg_prob = (prob[0] + (1 - prob[1])) / 2     
            predictions.append([team_names[team1], team_names[team2], avg_prob])


# Any now we can look at the results.

# In[ ]:


pd.DataFrame(predictions, columns=['Team1', 'Team2', 'Team1 Win Prob'])


# In[ ]:




