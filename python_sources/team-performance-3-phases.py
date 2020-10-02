#!/usr/bin/env python
# coding: utf-8

# Usually an T20 innings in cricket is divided into 3 phase:
# 
#  1. PowerPlay (1-6 overs): Field restrictions apply. Only 2 players are allowed beyond the 30 yard circle.
#  2. Middle overs(7-15) : Upto 5 fielders are allowed beyond the 30 yard circle.
#  3. Slog or death overs(15-20) : Field restrictions are as middle overs, but batting team tend to slog and score more runs at a higher strike rate.
# 
# In this notebook, I am trying to compare the batting & bowling performances of each team in these phases through metrics run rate, strike rate, economy rate,wickets lost etc.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# In[ ]:


# read the input files and look at the top few lines #
data_path = "../input/"
match_df = pd.read_csv(data_path+"matches.csv")
deliveries_df = pd.read_csv(data_path+"deliveries.csv")


# In[ ]:


match_df.head()


# In[ ]:


deliveries_df.head()


# Subset deliveries data frame to contain just first 6 overs

# In[ ]:


team_name_map = {'Mumbai Indians':'MI',                
'Royal Challengers Bangalore':'RCB' ,   
'Kings XI Punjab':'KXIP'    ,            
'Chennai Super Kings':'CSK' ,           
'Delhi Daredevils' :'DD'   ,           
'Kolkata Knight Riders'   :'KKR' ,      
'Rajasthan Royals'    :'RR'    ,      
'Deccan Chargers'    :'DC'   ,          
'Sunrisers Hyderabad' :'SRH' ,          
'Pune Warriors'       :'PWI' ,           
'Gujarat Lions'          :'GL',        
'Kochi Tuskers Kerala'      :'KT' ,   
'Rising Pune Supergiants':'RPS'}
deliveries_df['batting_team'] = deliveries_df['batting_team'].apply(lambda x: team_name_map[x])
deliveries_df['bowling_team'] = deliveries_df['bowling_team'].apply(lambda x: team_name_map[x])


# In[ ]:


deliveries_df['is_boundary'] = np.where(deliveries_df['batsman_runs'].isin([4,6]),1,0)
deliveries_df['is_wicket'] = np.where(deliveries_df['player_dismissed'].isnull(),0,1)
deliveries_df['is_bowler_wicket'] = np.where((deliveries_df['player_dismissed'].notnull())&(deliveries_df['dismissal_kind']!='run out'),1,0)
deliveries_df['phase'] = np.where(deliveries_df['over']<=6,'powerplay',np.where(deliveries_df['over']>15,'slog/death','middle'))


# In[ ]:


deliveries_df.columns


# In[ ]:


deliveries_df.head()


# In[ ]:


batting_grouped = deliveries_df.groupby(['batting_team','match_id','phase']).agg({'total_runs':sum,'ball':'count','is_wicket':sum,'is_boundary':sum,'over':pd.Series.nunique}).reset_index()


# In[ ]:


batting_grouped.head()


# In[ ]:


batting_grouped.columns = ['team','match','phase','runs_scored','balls_faced','wickets_lost','boundaries_scored','overs_faced']


# In[ ]:


batting_grouped['run_rate'] = batting_grouped['runs_scored']*1.0/batting_grouped['overs_faced']
batting_grouped['strike_rate'] = batting_grouped['runs_scored']*100.0/batting_grouped['balls_faced']


# In[ ]:


overall_run_rate = deliveries_df.groupby(['batting_team','match_id']).agg({'total_runs':sum,'over':pd.Series.nunique}).reset_index()


# In[ ]:


overall_run_rate.columns = ['team','match','runs_scored','overs_faced']
overall_run_rate.head()


# In[ ]:


overall_run_rate['match_run_rate'] = overall_run_rate['runs_scored']*1.0/overall_run_rate['overs_faced']
overall_avg_run_rate = overall_run_rate.groupby('team').agg({'match_run_rate':'mean','match':'count'}).reset_index()


# In[ ]:


agg = batting_grouped.groupby(['team','phase']).agg({'run_rate':'mean'}).reset_index()
table = agg.pivot(index='team', columns='phase', values='run_rate').reset_index()
table = table.merge(overall_avg_run_rate,on='team',how='left')


# In[ ]:


table = table[['team', 'powerplay','middle', 'slog/death', 'match_run_rate','match']]


# In[ ]:


table


# **RCB** has the highest average match run rate bonkers during last 5 overs. **CSK and KXIP** are the teams pacing their innings well by scoring more than 7.5 runs per over in all 3 phases.
# 
# Next, let us look at bowling performances of the teams in 3 phases.

# In[ ]:


bowling_grouped = deliveries_df.groupby(['bowling_team','match_id','phase']).agg({'total_runs':sum,'ball':'count','is_wicket':sum,'is_boundary':sum,'over':pd.Series.nunique}).reset_index()
bowling_grouped.columns = ['team','match','phase','runs_conceded','balls_bowled','wickets_taken','boundaries_conceded','overs_bowled']
bowling_grouped['econ_rate'] = bowling_grouped['runs_conceded']*1.0/bowling_grouped['overs_bowled']

overall_econ_rate = deliveries_df.groupby(['bowling_team','match_id']).agg({'total_runs':sum,'over':pd.Series.nunique}).reset_index()
overall_econ_rate.columns = ['team','match','runs_conceded','overs_bowled']
overall_econ_rate['match_econ_rate'] = overall_econ_rate['runs_conceded']*1.0/overall_econ_rate['overs_bowled']
overall_avg_econ_rate = overall_econ_rate.groupby('team').agg({'match_econ_rate':'mean','match':'count'}).reset_index()


# In[ ]:


agg = bowling_grouped.groupby(['team','phase']).agg({'econ_rate':'mean'}).reset_index()
table = agg.pivot(index='team', columns='phase', values='econ_rate').reset_index()
table = table.merge(overall_avg_econ_rate,on='team',how='left')
table = table[['team', 'powerplay','middle', 'slog/death', 'match_econ_rate','match']]
table


# If you look at teams with matches played more than 100, **KKR** interestingly has the lowest overall run rate but not topping any of the individual phases. **MI** have contain  opponents to lower score in power play giving away only 7 runs per over and also best team(malinga,bumrah etc.) in death overs conceding 8.8 runs per over.

# Next up, analyzing individual players for the 3 phases.. Let us see if we can come up with a team by selecting top players in these phases.

# Metrics to find best players in the 3 phases:
# 
#  1. **PowerPlay** : **Strikerate** for a batsmen (How quickly one scores runs) & **Economy** for a bowler(runs conceded per over)
#  2. **Middle overs** : **Dot ball %** for a batsmen (one needs to score at a steady rate) & **Strike rate** for a bowler (balls per wicket, need to pick up wickets to stop the steady flow)
#  3. **Slog/Death** : number of **boundaries** scored for a batsman and number of **boundaries** conceded for a bowler.

# #POWERPLAY

# In[ ]:


deliveries_df.columns


# In[ ]:


powerplay_df = deliveries_df[deliveries_df['phase']=='powerplay']


# In[ ]:


batsmen_powerplay_grouped = powerplay_df.groupby(['batsman','match_id']).agg({'batsman_runs':'sum','ball':'count'}).reset_index().rename(columns={'batsman_runs':'runs_scored','ball':'balls_faced'})
batsmen_powerplay_grouped['strikerate'] = batsmen_powerplay_grouped['runs_scored']*100.0/batsmen_powerplay_grouped['balls_faced']


# In[ ]:


agg = batsmen_powerplay_grouped.groupby('batsman').agg({'strikerate':'mean','balls_faced':'sum'}).reset_index().rename(columns={'strikerate':'avg_sr','balls_faced':'total_balls_faced'})
agg = agg[agg['total_balls_faced']>=300]
agg = agg.sort_values('avg_sr',ascending=False)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


g = sns.barplot(x='batsman',y='avg_sr',data=agg.iloc[0:10,]);
fig = g.get_figure()
fig.set_size_inches(12,10)
fig.suptitle("Average Strikerate Top 10 Batsmen in Powerplay(played more than 300 balls)")
fig.tight_layout()
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height),ha="center") 


# In[ ]:


bowler_powerplay_grouped = powerplay_df.groupby(['bowler','match_id']).agg({'total_runs':'sum','over':pd.Series.nunique}).reset_index().rename(columns={'total_runs':'runs_conceded','over':'overs_bowled'})
bowler_powerplay_grouped['economyrate'] = bowler_powerplay_grouped['runs_conceded']*1.0/bowler_powerplay_grouped['overs_bowled']


# In[ ]:


agg = bowler_powerplay_grouped.groupby('bowler').agg({'economyrate':'mean','overs_bowled':'sum'}).reset_index().rename(columns={'economyrate':'avg_er','overs_bowled':'num_overs_bowled'})
agg = agg[agg['num_overs_bowled']>=50]
agg = agg.sort_values('avg_er',ascending=True)


# In[ ]:


g = sns.barplot(x='bowler',y='avg_er',data=agg.iloc[0:10,]);
fig = g.get_figure()
fig.set_size_inches(12,10)
fig.suptitle("Top 10 Bowlers in Powerplay(bowled more than 50 overs)")
fig.tight_layout()
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,height + 0.1,'{:1.2f}'.format(height),ha="center") 


# ### Top 3 Batsmen for  Powerplay
# 
#  - **Virender Sehwag**(India & KXIP) with 126.34 strikerate
#  - **Faf du Plessis** (SA & CSK) with 125.03 strikerate
#  - **Chris Gayle** (WI & RCB) with 122.79 strikerate
# 
# ### Top 3 Bowlers for  Powerplay
# 
#  - **Sunil Narine**(WI & KKR) with 5.53 economy rate
#  - **Lasith Malinga** (SL & MI) with 6.01 economy rate
#  - **Bhuvaneshwar Kumar** (India & SRH) with 6.38 economy rate
# 
# The interesting thing is that Sunil narine , an off spinner with lowest economy rate in the power play.

# # Middle Overs

# In[ ]:


middle_df = deliveries_df[deliveries_df['phase']=='middle']
middle_df['is_dot_ball'] = middle_df.apply(lambda row: 1 if row['batsman_runs']==0 & row['extra_runs']==0 else 0,axis=1)


# In[ ]:


batsmen_middle_grouped = middle_df.groupby(['batsman','match_id']).agg({'is_dot_ball':sum,'ball':'count'}).reset_index().rename(columns={'is_dot_ball':'dot_balls','ball':'balls_faced'})
batsmen_middle_grouped['dotball%'] = batsmen_middle_grouped['dot_balls']*100.0/batsmen_middle_grouped['balls_faced']


# In[ ]:


agg = batsmen_middle_grouped.groupby('batsman').agg({'dotball%':'mean','balls_faced':'sum'}).reset_index().rename(columns={'dotball%':'avg_dot_ball_%','balls_faced':'total_balls_faced'})
agg = agg[agg['total_balls_faced']>=600]
agg = agg.sort_values('avg_dot_ball_%',ascending=True)


# In[ ]:


g = sns.barplot(x='batsman',y='avg_dot_ball_%',data=agg.iloc[0:10,]);
fig = g.get_figure()
fig.set_size_inches(12,10)
fig.suptitle("Average Dot ball % Top 10 Batsmen in middle overs(played more than 600 balls)")
fig.tight_layout()
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,height + 0.3,'{:1.2f}'.format(height),ha="center") 


# In[ ]:


agg.tail(10)


# The top 3 players with minimum dot ball % are **Rahane, Warner and Hussey** . 6 of the 10 players are openers and none of them featured in players with top strike rate in power play except for **Sehwag**. 
# 
# The interesting find lies not in top 10 but bottom 10 players in this category. **Yuvraj, Gayle, Gangly,Watson and Dhawan** etc.. This list consist blend of players topping the powerplay category and players who just come to play in middle overs. Players either run out of gas after power play or take their time to really slog in the final overs.
# 

# In[ ]:


bowler_middle_grouped = middle_df.groupby(['bowler','match_id']).agg({'is_bowler_wicket':'sum','over':pd.Series.nunique}).reset_index().rename(columns={'is_bowler_wicket':'wickets_taken','over':'overs_bowled'})


# In[ ]:


agg = bowler_middle_grouped.groupby('bowler').agg({'wickets_taken':'sum','overs_bowled':'sum'}).reset_index().rename(columns={'strikerate':'avg_sr','overs_bowled':'num_overs_bowled'})
agg = agg[agg['num_overs_bowled']>=100]
agg['strikerate'] = agg['num_overs_bowled']*6.0/agg['wickets_taken']
agg = agg.sort_values('strikerate',ascending=True)


# In[ ]:


agg.tail(10)


# In[ ]:


g = sns.barplot(x='bowler',y='strikerate',data=agg.iloc[0:10,]);
fig = g.get_figure()
fig.set_size_inches(12,10)
fig.suptitle("Top 10 Bowlers in Middle overs(bowled more than 100 overs)")
fig.tight_layout()
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,height + 0.1,'{:1.2f}'.format(height),ha="center") 


# No wonder this category is dominated by spinners(9 of 10) all of them are Indian players(expected), as it is normal for teams to use spinners after power play(with no field restrictions). The top 3 bowlers are **chahal(amazing strikerate of a wicket for every 3rd over),Karn sharma and axar patel**. 
# 
# All of them are young talents and interestingly looking at worst 10 we see some big names **Kumble,Muralitharan and Kallis** showing that T20 requires different bowling strategies than conventional.

# ### Top 3 Batsmen for  Middle overs
# 
#  - **Ajinkiya Rahane**(IND & RR) with 33.1 dot ball %
#  - **David Warner** (AUS & SRH) with  33.2 dot ball %
#  - **Michael Hussey** (AUS & CSK) with  33.3 dot ball %
# 
# ### Top 3 Bowlers for  Middle overs
# 
#  - **Yuvendra Chahal**(IND & RCB) with a wicket every 18 balls
#  - **Karn Sharma** (IND & MI) with a wicket every 21 balls
#  - **Axar Patel** (IND & KXIP) with a wicket every 21 balls

# # Slog/Death Overs

# In[ ]:


slog_df = deliveries_df[deliveries_df['phase']=='slog/death']


# In[ ]:


agg = slog_df.groupby(['batsman']).agg({'is_boundary':sum,'ball':'count'}).reset_index().rename(columns={'is_boundary':'num_boundaries','ball':'balls_faced'})
agg = agg[agg['balls_faced']>=200]
agg['big_shot_rate'] = agg['balls_faced']/agg['num_boundaries']
agg = agg.sort_values('big_shot_rate',ascending=True)


# In[ ]:


agg.tail(10)


# In[ ]:


g = sns.barplot(x='batsman',y='big_shot_rate',data=agg.iloc[0:10,]);
fig = g.get_figure()
fig.set_size_inches(12,10)
fig.suptitle("Top 10 Batsmen in slog/death overs(180 balls or more)")
fig.tight_layout()
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,height + 0.1,'{:1.2f}'.format(height),ha="center") 


# Surprisingly the top players in this category are **ABD,Kohli and Rohit sharma** are either openers or one-down batsmen. This shows that they play a long innings making themselves most valuable players in the team. **Yuvraj,Dhoni and Miller** are the usual suspects since they are known for their pinch-hitting and finisher roles.

# In[ ]:


agg = slog_df.groupby(['bowler']).agg({'is_boundary':sum,'ball':'count'}).reset_index().rename(columns={'is_boundary':'num_boundaries','ball':'balls_bowled'})
agg = agg[agg['balls_bowled']>=200]
agg['containing_rate'] = agg['balls_bowled']/agg['num_boundaries']
agg = agg.sort_values('containing_rate',ascending=False)


# In[ ]:


agg.tail(10)


# In[ ]:


g = sns.barplot(x='bowler',y='containing_rate',data=agg.iloc[0:10,]);
fig = g.get_figure()
fig.set_size_inches(12,10)
fig.suptitle("Top 10 Bowlers in slog/death overs(40 overs or more)")
fig.tight_layout()
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,height + 0.1,'{:1.2f}'.format(height),ha="center") 


# ### Top 3 Batsmen for  slog/death overs
# 
#  - **ABD**(SA & RCB) with a boundary every 3.2 balls
#  - **Virat Kohli** (IND & RCB) with a boundary every 3.8 balls
#  - **Rohit Sharma** (IND & MI) with a boundary every 3.8 balls
# 
# ### Top 3 Bowlers for  Middle overs
# 
#  - **Lasith Malinga**(SL & MI) conceding boundary only every 9 balls
#  - **Ravichandran Ashwin** (IND & CSK) conceding boundary only every 8 balls
#  - **Sunil Narine** (WI & KKR) conceding boundary only every 8 balls

# **Overall when you break an inning into 3 phases and look at the performance, in the batting side you can't keep ABD and Kohli out of your team and in bowling side Sunil Narine and Lasith Malinga seems to lead the pack** 

# In[ ]:




