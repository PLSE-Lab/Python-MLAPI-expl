#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pylab 
import scipy.stats as stats


# In[ ]:


# Read data
players = pd.read_csv('../input/fifa19/data.csv')
players.info()


# In[ ]:


# Extracting players from teams in the premier league
teams = ['Arsenal', 'Bournemouth', 'Brighton & Hove Albion', 'Burnley', 'Cardiff City', 'Chelsea', 
         'Crystal Palace', 'Everton', 'Fulham', 'Huddersfield Town', 'Leicester City', 'Liverpool',
         'Manchester City', 'Manchester United', 'Newcastle United', 'Southampton', 
         'Tottenham Hotspur', 'Watford', 'West Ham United', 'Wolverhampton Wanderers']
T = 20 # Number of teams


# In[ ]:


# Create a list of teams by players
teamsByPlayers = []
for team in teams:
    teamsByPlayers.append((team, players.loc[players['Club']==team]))
# Remove player and regular team data
del players, teams


# In[ ]:


# Define defensive and attacking positions
defensivePlayerPositions = ['LCM','CM','RCM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB']
attackingPlayerPositions = ['LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM','RCM','RM']

# Define traits by positions
traitsByPosition = {
    'GK':   ['GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes'],
    'def':  ['ShortPassing','LongPassing','BallControl','Agility','Reactions','Stamina',
             'Strength','Aggression','Interceptions','Positioning','Vision',
             'Marking','StandingTackle','SlidingTackle','HeadingAccuracy','Jumping'],
    'att':  ['Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling',
             'Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed',
             'Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength',
             'LongShots','Vision','Penalties','Composure']
}

# Give the team an attack and a defence rating
def getTeamRating(players):
    # Pick best 23 players
    players = players.sort_values(by=['Overall'], ascending=False).head(23)
    # Get ratings
    GKRating        = np.mean(np.mean(players.loc[players['Position'] == 'GK'][traitsByPosition['GK']]))
    defensiveRating = np.mean(np.mean(players.loc[players['Position'].isin(defensivePlayerPositions)][traitsByPosition['def']]))
    attackingRating = np.mean(np.mean(players.loc[players['Position'].isin(attackingPlayerPositions)][traitsByPosition['att']]))
    # Adjust defensive rating with GK rating
    GKCount  = len(players.loc[players['Position'] == 'GK'])
    defCount = len(players.loc[players['Position'].isin(defensivePlayerPositions)])
    defensiveRating = (GKCount * GKRating + defCount * defensiveRating)/(GKCount + defCount)
    return [defensiveRating, attackingRating]


# In[ ]:


# Create list of teams by ratings
teamsByRating = []
for team, players in teamsByPlayers:
    teamsByRating.append((team, getTeamRating(players)))


# In[ ]:


# Normalize ratings
for i in range(2):
    m = np.mean([x[1][i] for x in teamsByRating])
    s = np.std([x[1][i] for x in teamsByRating])
    for team in teamsByRating:
        team[1][i] -= m
        team[1][i] /= s


# In[ ]:


# 10 best rated defenses
teamsByRating.sort(key=lambda x: x[1][0], reverse=True)
teamsByRating[:10]


# In[ ]:


measurements = np.array([x[1][0] for x in teamsByRating])
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


plt.hist(measurements, bins=6) # 5 approx sqrt(20)


# In[ ]:


# 10 best rated attacks
teamsByRating.sort(key=lambda x: x[1][1], reverse=True)
teamsByRating[:10]


# In[ ]:


measurements = np.array([x[1][1] for x in teamsByRating])
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


plt.hist(measurements, bins=5) # 5 approx sqrt(20)


# In[ ]:


# Teams by rating
teamsByRating.sort(key=lambda x: x[1][0] + x[1][1], reverse=True)
teamsByRating


# In[ ]:


# These ratings seem quite appropriate given the performances of these teams in the season before FIFA 19 was released.


# In[ ]:


# Changing teamsByRating into a map for faster access:
teamMap = {}
for team, rating in teamsByRating:
    teamMap[team] = rating


# ## The Model
# 
# ### Using Poisson Processes
# 
# It is a well known fact that goals scored in football follow a Poisson process. To replicate such a process, we require some notion of average goals scored by a team. However, if we use a particular season to obtain this data for each team individually, it wouldn't be representative of the team skill, but rather their performance from that season. As a result, we need to use some sort of generic notion of average goals scored by a team, and use the team ratings obtained from FIFA (which obviously can be chosen in different, and likely better ways than how I've done it above, but this makes for a good place to start) to modify this stat. Here, we will use the average number of goals scored by a home team and the average number of goals scored by an away team and use some notion of their attacking and defensive skill ratings to modify this average for the particular game. We have:
# $$ \mu_{home}, \sigma_{home} $$
# for the home team, and
# $$ \mu_{away}, \sigma_{away} $$
# for the away team obtained from the 2017-18 season. 
# 
# ### Adjusting with Team Skill
# 
# We have skill ratings $\alpha_i, \beta_i$ (offensive and defensive respectively) for the $i^{th}$ team in the league such that:
# $$\alpha\sim N(0,1)$$
# and
# $$\beta\sim N(0,1)$$
# obtained from FIFA's player data for each team.
# 
# As such, if we consider a game featuring teams $T_i$ and $T_j$, we require $\mu_{home, i} = f(\mu_{home}, \alpha_i, \beta_j)$ (number of goals score by the $i^{th}$ team at home against the visitor $T_j$) and $\mu_{away, j} = g(\mu_{away}, \alpha_j, \beta_i)$ (number of goals score by the $j^{th}$ team away against the hosts $T_i$) such that:
# 
# $$\mu_{home} = \frac{1}{n}\sum_i E_j(\mu_{home, i})$$
# and
# $$\mu_{away} = \frac{1}{n}\sum_j E_i(\mu_{away, j})$$
# 
# For a basic start, we can enforce $f=g$. Now, we have the following restrictions on $f$:
# 
# $$ \lim_{\beta\to\infty} f(\mu, \alpha, \beta) = 0,  \lim_{\alpha\to\infty} f(\mu, \alpha, \beta) = \infty $$
# 
# Furthermore, we want:
# 
# $$ E(f) \approx \mu $$
# 
# The following $f$ satisfies these criteria (again, there might be better ways to deal with this):
# 
# $$ f := \frac{\mu (1+k)^{\alpha - \beta}}{E\{(1+k)^{\alpha-\beta}\}} $$
# 
# for some small $k>0$.
# 
# As a result, a game $T_i$ vs $T_j$ (where $i$ is at home) can be played out by first obtaining:
# 
# $$\mu_i = f(\mu_{home}, \alpha_i, \beta_j), \mu_j = f(\mu_{away}, \alpha_j, \beta_i) $$
# 
# And then drawing the number of goals the teams score from a Poisson distribution with means $\mu_i$ and $\mu_j$ respectively.

# In[ ]:


# Getting mu_home and mu_away from the 2017/18 season results.
df = pd.read_csv('../input/201718-premier-league-matches/season-1718.csv')
mu_home = np.mean(df['FTHG'])
mu_away = np.mean(df['FTAG'])
print(mu_home, mu_away)
del df


# In[ ]:


# Setting k -- the amount of effect skill has
k = 0.25

# Getting adjusted mu value:
def f(mu, a, b):
    return (mu) * ((1+k) ** (a - b))

# Obtaining E
E = np.mean([f((mu_home + mu_away)/2, teamsByRating[i][1][1], teamsByRating[j][1][0]) for i in range(20) for j in range(20) if i != j for k in range(10000)])/((mu_home + mu_away)/2)

# Redefining adjusted mu value:
def f(mu, a, b):
    return (mu) * ((1+k) ** (a - b))/E

E = np.mean([f((mu_home + mu_away)/2, teamsByRating[i][1][1], teamsByRating[j][1][0]) for i in range(20) for j in range(20) if i != j for k in range(10000)])/((mu_home + mu_away)/2)
print(E) # Should be very close to 1

del teamsByRating


# In[ ]:


# Simulate game given home and away team
def getHomeAwayLambda(home, away):
    bi, ai = teamMap[home]
    bj, aj = teamMap[away]
    lam_home = f(mu_home, ai, bj)
    lam_away = f(mu_away, aj, bi)
    return lam_home, lam_away

def simulateGame(home, away):
    lam_home, lam_away = getHomeAwayLambda(home, away)
    fthg = np.random.poisson(lam=lam_home)
    ftag = np.random.poisson(lam=lam_away)
    return fthg, ftag


# In[ ]:


# Get Fixture List
fixtures = pd.read_csv('../input/fixtures/fixtures.csv')
fixtures.info()


# In[ ]:


# Simulate season
def simulateSeason(fixtures):
    season = pd.DataFrame(columns=['HomeTeam', 'FTHG', 'FTAG', 'AwayTeam'])
    for index, row in fixtures.iterrows():
        home, away = row['HOME TEAM'], row['AWAY TEAM']
        fthg, ftag = simulateGame(home, away)
        season = season.append({'HomeTeam': home, 'FTHG': fthg, 'FTAG': ftag, 'AwayTeam': away}, ignore_index=True)
    return season


# In[ ]:


# Create table from season
def getLeagueTable(season):
    columns = ['P', 'W', 'D', 'L', 'GFH', 'GAH', 'GFA', 'GAA', 'PTS']
    index = []
    for i, game in season.iterrows():
        if game['HomeTeam'] not in index:
            index.append(game['HomeTeam'])
        if len(index) == T:
            break
    table = pd.DataFrame(columns=columns, index=index)
    table[:] = 0
    for i, game in season.iterrows():
        # Update Played
        table.loc[game['HomeTeam'], 'P'] += 1
        table.loc[game['AwayTeam'], 'P'] += 1
        # Update Goals
        table.loc[game['HomeTeam'], 'GFH'] += game['FTHG']
        table.loc[game['HomeTeam'], 'GAH'] += game['FTAG']
        table.loc[game['AwayTeam'], 'GFA'] += game['FTAG']
        table.loc[game['AwayTeam'], 'GAA'] += game['FTHG']
        # Update Points
        if game['FTHG'] > game['FTAG']:
            table.loc[game['HomeTeam'], 'W'] += 1
            table.loc[game['AwayTeam'], 'L'] += 1
        elif game['FTHG'] < game['FTAG']:
            table.loc[game['HomeTeam'], 'L'] += 1
            table.loc[game['AwayTeam'], 'W'] += 1
        else:
            table.loc[game['HomeTeam'], 'D'] += 1
            table.loc[game['AwayTeam'], 'D'] += 1
    table['GF'] = table['GFH'] + table['GFA']
    table['GA'] = table['GAH'] + table['GAA']
    table['GD'] = table['GF'] - table['GA']
    table['PTS'] = table['D'] + 3 * table['W']
    table = table.sort_values(by=['PTS', 'GD'], ascending=False)
    return table


# In[ ]:


s = simulateSeason(fixtures)
table = getLeagueTable(s)
table


# In[ ]:


# Simulate N seasons
def simulateNSeasons(N):
    print("Initializing")
    # Initialize results
    results = {}
    for team in teamMap:
        results[team] = [0] * T
    # Simulate seasons
    for iteration in range(N):
        print("Running simulation %d" % iteration)
        season = simulateSeason(fixtures)
        table = getLeagueTable(season)
        for i in range(T):
            results[table.iloc[i].name][i] += 1
    # Create result matrix
    print("Creating result matrix")
    columns = [str(i+1) for i in range(T)]
    index = []
    for team in teamMap:
        index.append(team)
    resultMatrix = pd.DataFrame(columns=columns, index=index)
    resultMatrix[:] = 0
    for team in results:
        for i in range(T):
            resultMatrix.loc[team, str(i+1)] += results[team][i]
    # Get average position
    resultMatrix['Avg'] = [(i+1) * resultMatrix[str(i+1)] / N for i in range(T)]
    resultMatrix['Avg'] = 0
    for team in teamMap:
        s = 0
        for i in range(0, T):
            s += resultMatrix.loc[team, str(i+1)] * (i+1)
        resultMatrix.loc[team, 'Avg'] = s/N
    resultMatrix = resultMatrix.sort_values(by=['Avg'], ascending=True)
    return resultMatrix


# In[ ]:


simulateNSeasons(100)


# In[ ]:




