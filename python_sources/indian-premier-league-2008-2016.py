#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


deliveries = pd.read_csv('../input/deliveries.csv')
deliveries.head(1)


# In[ ]:


deliveries.groupby(['match_id', 'inning']).agg({'total_runs': 'sum'})[:6]


# In[ ]:


matches = pd.read_csv('../input/matches.csv')
matches.head(1)


# ### How many matches have been played in IPL so far by season?

# In[ ]:


matches.hist(column='season', figsize=(9,6),bins=20)  


# In[ ]:


matches['city'].value_counts().plot(kind='barh', figsize=(10,8), rot=0)


# ### Merging matches and deliveries dataframes to create a new dataframe called ipl 

# In[ ]:


ipl = matches[['id', 'season']].merge(deliveries, left_on = 'id', right_on ='match_id').drop('match_id', axis = 1)


# In[ ]:


ipl.head(1)


# In[ ]:


ipl.shape


# In[ ]:


ipl.columns


# #### How many times has the run 1,2,3,4,6 scored in IPL?

# In[ ]:


ipl.total_runs.value_counts()


# #### There were 56189 times 1 run was scored, 5784 times sixer were scored.

# ### 1) How many matches the teams have won in each season?
# ### 2) Plot the total number of matches won in all IPL

# In[ ]:


matches.groupby('season').winner.value_counts()


# In[ ]:


match_winners = matches.winner.value_counts()
fig, ax = plt.subplots(figsize=(8,7))
explode = (0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4)
ax.pie(match_winners, labels = None, autopct='%1.1f%%', startangle=90, shadow = True, explode = explode)
ax.legend(bbox_to_anchor=(1,1), labels=match_winners.index)


# ### [](http://) What are the total runs scored in each innings in all IPL matches?

# In[ ]:


ipl.groupby(['inning']).agg({'total_runs': 'sum'})


# ### How many balls were bowled in each inning?

# In[ ]:


ipl.groupby(['inning']).agg({'ball': 'count'})


# ### What is the total number of runs scored against balls bowled in each innings per season in all matches?

# In[ ]:


ipl.groupby(['season','inning']).agg({'total_runs': 'sum', 'ball': 'count'})


# In[ ]:


ipl.groupby(['season','inning']).agg({'total_runs': 'sum'}).plot(kind='barh', figsize=(10,8))
plt.xlabel('Total Runs scored by all teams')
plt.ylabel('Season and Innings')


# ### What is the total runs scored per season?

# In[ ]:


ipl.groupby(['season']).agg({'total_runs': 'sum'}).plot(kind='bar')


# In[ ]:


y = ipl.groupby(['season']).agg({'total_runs': 'sum', 'ball': 'count'})
x = matches.groupby(['season']).agg({'id': 'count'})


# In[ ]:


x2 = x.reset_index()
x2


# In[ ]:


y2= y.reset_index()
y2


# In[ ]:


iplz = pd.merge(x2,y2, how='inner', on='season')
ipl2=iplz.set_index('season')
ipl2


# In[ ]:


ipl2.plot(kind ='barh', colormap = 'plasma')


# ### 1) What is the average runs scored per match by season?
# ### 2) What is the average balls bowled per match by season?
# ### 3) What is average runs scored against each ball bowled per season?
# ### 4) Plot

# In[ ]:


t = ipl2['total_runs']
b = ipl2['ball']
n = ipl2['id']
tn = (t.T / n.T).T
bn = (b.T / n.T).T
tb = (t.T / b.T).T
z = pd.DataFrame([n, tn,bn,tb])
z.index = ['No.of matches', 'Average Runs per match', 'Average balls bowled per match', 'Average runs per ball']
z.T


# In[ ]:


z.T.plot(kind='bar', figsize = (12,10), colormap = 'coolwarm')
plt.xlabel('Season')
plt.ylabel('Average')
plt.legend(loc=9,ncol=4)


# #### Alternately, visualise thru a subplot

# In[ ]:


plt.figure(figsize=(12,8))
t = ipl2['total_runs']
b = ipl2['ball']
n = ipl2['id']
tn = (t.T / n.T).T
bn = (b.T / n.T).T
tb = (t.T / b.T).T
ax = plt.subplot(311) 
plt.plot( tn, 'b.--',)
ax.set_title("Average Runs scored per match")
ax1 = plt.subplot(312)
plt.plot( bn, 'r.--')
ax1.set_title("Average Balls bowled per match")
ax2 = plt.subplot(313)
plt.plot( tb, 'g-.')
ax2.set_title("Average Runs scored per Ball")
plt.subplots_adjust(top=2.0)
plt.tight_layout()


# ### What is Virat Kohli's strike rate in IPL matches over seasons?[](http://)

# In[ ]:


ak = ipl[ipl.batsman.str.lower().str.contains('kohli')].groupby(['season'])['total_runs'].count()
bk = ipl[ipl.batsman.str.lower().str.contains('kohli')].groupby(['season'])['total_runs'].sum()
ck = pd.concat([ak, bk], axis=1)
kohli_strikerate = (bk.T / ak.T*100).T
kohli_strikerate.plot('bar', figsize=(10,8))
plt.xlabel('Season')
plt.ylabel('Virat Kohli Strike Rate (Runs scored per ball)')
plt.title('Virat Kohli in IPL')


# In[ ]:


deliveries[deliveries["batsman"] == "V Kohli"]["batsman_runs"].value_counts().plot(kind="bar")
plt.title('V Kohli')
plt.ylabel('No of times')
plt.xlabel('Runs')


# ### List top 10 batsman who have faced number of balls in IPL?

# In[ ]:


ipl['batsman'].value_counts()[:10]


# In[ ]:





# ### Bowler is 'R Ashwin', How many runs were scored of his bowling by opponent teams in IPL

# In[ ]:


# P Kumar's teamwise - runs scored by teams of his bowling
Ash = deliveries[deliveries['bowler'] == 'R Ashwin']  # inning 2
bowler_Ash = Ash.groupby('batting_team')['total_runs'].sum()
bowler_Ash


# In[ ]:


bowler_Ash.sum()


# #### Total runs given away by R Ashwin is entire IPL is 2552 runs, inning-wise break-up is given below:

# In[ ]:


inning_Ash = Ash.groupby('inning')['total_runs'].sum()
inning_Ash


# #### Break-up of number of times each run was scored of R Ashwin's bowling in each inning.

# In[ ]:


Run_Ashwin = Ash.groupby('inning')['total_runs'].value_counts()
Run_Ashwin


# #### Dismissal Kind of R Ashwin's bowling

# In[ ]:


dismissal_Ash = Ash.groupby('dismissal_kind')['player_dismissed'].count()
dismissal_Ash.plot('bar', rot =50)
plt.ylabel('Number of times')
plt.title('R Ashwin dismissal kind')


# ### How many matches were played and won by teams in all IPL?

# In[ ]:


all_teams = matches['team1'].unique().tolist() + matches['team2'].unique().tolist()
all_teams = list(set(all_teams))

team_names =[]
played_count = []
won_count = []
for team in all_teams:
    team_names.append(team)
    played = matches[(matches['team1'] == team) | (matches['team2'] == team)].shape[0]
    won = matches[matches['winner'] == team].shape[0]
    
    played_count.append(played)
    won_count.append(won)

data_dict = {
    'team': team_names,
    'played': played_count,
    'won': won_count,   
}

df_played_won = pd.DataFrame(data_dict)
team_won = df_played_won.set_index('team')
team_won.plot(kind='barh', figsize=(10,10))
plt.xlabel('No of times')


# In[ ]:


season_wins = pd.crosstab(matches.winner, matches.season, margins=True)
season_wins


# In[ ]:


matches.toss_winner.value_counts()


# In[ ]:


matches.toss_decision.value_counts()


# In[ ]:


matches.groupby(['season','toss_decision'])['id'].count().plot('barh', figsize=(10,8))


# In[ ]:


matches.groupby(['season','result'])['id'].count()


# In[ ]:


matches.groupby(['toss_winner', 'toss_decision']).id.count()


# In[ ]:


matches.groupby([ 'toss_decision', 'toss_winner']).id.count()


# In[ ]:


matches.groupby('winner')['win_by_runs'].agg(np.max).plot(kind = 'barh', figsize=(10,8))


# In[ ]:


matches.pivot_table( columns='winner', values='win_by_runs', aggfunc='max').T


# In[ ]:


over = deliveries.groupby('over', as_index=True).agg({"total_runs": "sum"})
over.plot(kind='bar', figsize=(12,8));


# In[ ]:


over = deliveries.groupby('over', as_index=True).agg({"extra_runs": "sum"})
over.plot(kind='bar', figsize=(12,8));


# In[ ]:


overm = deliveries.groupby('ball', as_index=True).agg({"total_runs": "sum"})
overm.plot(kind='barh', figsize=(12,8));


# In[ ]:


ball_df = deliveries['total_runs'].groupby([deliveries['inning'], deliveries['ball']]).mean()
ball_df


# In[ ]:


tot = ipl.groupby(['batting_team'])['total_runs']
tot.sum().plot('bar')


# In[ ]:


ipl.groupby(['batting_team','season','inning']).agg({'total_runs': [ 'max', sum],     
                                     'extra_runs': 'sum',
                                                'batsman_runs': 'sum',
                                     'ball': ['count', 'nunique']}).head(9)


# In[ ]:


ipl.groupby(['batsman']).batsman_runs.sum().sort_values(ascending = False).head(10)


# In[ ]:


grp = deliveries.groupby('batsman')['batsman_runs'].sum()
grp.reset_index().sort_values(by=['batsman_runs'], ascending = False).head()


# In[ ]:


grp1 = deliveries.groupby(['bowling_team','bowler'])['player_dismissed'].count()
grp1.reset_index().sort_values(by=['player_dismissed'], ascending = False)[:5]


# In[ ]:


ipl.groupby(['batsman', 'inning']).batsman_runs.sum().sort_values(ascending = False).head(10)


# #### times the batsman was dismissed

# In[ ]:


ipl.player_dismissed.value_counts().head()


# #### times the fielder has taken catches

# In[ ]:


ipl.fielder.value_counts().head()


# #### dismissals type and count

# In[ ]:


ipl.dismissal_kind.value_counts().head()


# In[ ]:


deliveries.batsman_runs.value_counts().plot('bar')
plt.xticks(np.arange(0,7, 1), rotation = 40);
plt.yticks(np.arange(0,60000, 4000));


# In[ ]:


deliveries.groupby(['inning','batting_team']).total_runs.sum()


# In[ ]:




