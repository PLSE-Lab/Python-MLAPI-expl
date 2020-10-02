#!/usr/bin/env python
# coding: utf-8

# <font size="5"><b><center>CS:GO Professional Matches Analysis</center></b></font>
# <center><b>Dataset: </b>CS:GO Professional Matches - <b>Date: </b>March 26, 2020</center>

# # Index
# 1. [Introduction](#Introduction)<br>
# 2. [Preparation](#Preparation)<br>
# 3. [Results DataFrame](#Results-DataFrame)<br>
# 3.1. [Distribution of scores](#Distribution-of-scores)<br>
# 3.2. [Most CT-sided map](#Most-CT-sided-map)<br>
# 3.3. [Maps played per period](#Maps-played-per-period)<br>
# 3.4. [Best teams on each map](#Best-teams-on-each-map)<br>
# 4. [Economy DataFrame](#Economy-DataFrame)<br>
# 4.1. [Round victory probability by equipment value](#Round-victory-probability-by-equipment-value)<br>
# 4.2. [Pistols ranking](#Pistols-ranking)<br>
# 5. [Players DataFrame](#Players-DataFrame)<br>
# 5.1.  [Players ranking by map](#Players-ranking-by-map)<br>
# 5.2. [Players ranking all maps](#Players-ranking-all-maps)<br>
# 5.3. [K/D Graph](#K/D-Graph)<br>
# 6. [Picks DataFrame](#Picks-DataFrame)<br>
# 7. [Further exploration and predictions](#Further-exploration-and-predictions)<br>

# # Introduction

# <b>Background:</b> The data used in this notebook was collected at https://www.hltv.org/results. It was scraped off using the libraries 'requests' and 'BeautifulSoup'.
# 
# I haven't set any specific goal for this analysis, the idea was just to explore the data and see what information I could get. Also, I tried to avoid computing statistics that are easily obtained in e-sports websites (team victory percentage, average k/d ratio per player, etc.) and instead focused on rankings and analysis with temporal aspects.
# 
# This is my first public kernel, so don't hesitate in giving me corrections and suggestions.
# 
# Let's get started!

# # The Game

# This section is copied from the Dataset description.
# 
# Counter-Strike Global Offensive is a game released in 2012, as a sequel to Counter-Strike Source (released in 2004), which is itself a sequel to the original Counter-Strike (released in 2000). The game's longevity is primarily caused by its competitive approach and vibrant professional scene. This longevity has shown in numbers recently, as CS:GO reached in March its all-time high concurrent weekly players (1.1M players), making it the most played game on Steam, 7 years after it was launched.
# 
# The game retains the same gameplay concepts since its first version, which include a Terrorist side (T) that is tasked to plant a bomb and have it detonate, and a Counter-Terrorist side (CT) that is tasked to defuse the bomb or prevent it from being planted. Both teams can also win a round by eliminating all players on the opposing team before the bomb is planted.
# 
# A standard game of Counter-Strike is a best of 30 rounds, the winning team being the first to win 16 rounds. The 30 rounds are played in two halves of 15 on each side of the map, with a round time limit of 1 minute 55 seconds, plus 40 seconds after the bomb is planted.
# 
# In case both teams draw at the 30th round on 15x15, 6 more rounds are added-on, which constitutes overtime. The overtime ends if a team wins 4 out of 6 rounds. If both teams win 3 rounds in overtime, another overtime of 6 rounds is played, and the process might repeat indefinitely until one team wins it.
# 
# There are 7 maps in the map pool that are available to be played competitively at any given time. Maps are removed and added frequently for updates and revamps, as to not make the game stale. Matches are normally played as a 'bo3' (Best of 3) maps, with less important matches played in a 'bo1' fashion and finals often played as 'bo5's.
# 
# There is a money management side to rounds in Counter-Strike. This side is detailed in the 'Economy DataFrame' topic.

# # Preparation

# Importing libraries and loading the tables.

# In[ ]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from os import listdir

pd.set_option('display.max_columns',100)

listdir('../input/csgo-professional-matches/')


# In[ ]:


base_dir = '../input/csgo-professional-matches/'

results_df = pd.read_csv(base_dir+'results.csv',low_memory=False)
picks_df = pd.read_csv(base_dir+'picks.csv',low_memory=False)
economy_df = pd.read_csv(base_dir+'economy.csv',low_memory=False)
players_df = pd.read_csv(base_dir+'players.csv',low_memory=False)


# The first rows of each table are presented below. The data is split into 4 tables that store data related to:
# 
#  - <b>results_df:</b>   &nbsp;  map scores and team rankings
# 
#  - <b>picks_df:</b> &nbsp;  order of map picks and vetos in the map selection process.
# 
#  - <b>economy_df:</b>  &nbsp;  round start equipment value for all rounds played
# 
#  - <b>players_df:</b> &nbsp;   individual performances of players in each map.
#  
# Values stored in 'event_id' and 'match_id' columns are unique for each match and event and shared between dataframes, so these columns can be used as keys to merge data between dataframes.
# 
# It is necessary to note that the rows in 'results_df' and 'economy_df' store data for each map played in a match, while the rows in 'picks_df' and 'players_df' store data for the entire match.

# In[ ]:


results_df.head()


# In[ ]:


picks_df.head()


# In[ ]:


economy_df.head()


# In[ ]:


players_df.head()


# The data collected has data from all professional CS:GO matches, including matches from relatively unknown teams. For that reason, we will be limiting the datasets to the matches played between the top 30 teams in the HLTV rankings.

# In[ ]:


min_rank = 30
results_df = results_df[(results_df.rank_1<min_rank)&(results_df.rank_2<min_rank)]

picks_df     = picks_df  [picks_df  .match_id.isin(results_df.match_id.unique())]
economy_df   = economy_df[economy_df.match_id.isin(results_df.match_id.unique())]
players_df   = players_df[players_df.match_id.isin(results_df.match_id.unique())]


# # Results DataFrame

# ## Distribution of scores

# In[ ]:


winner_1 = results_df[results_df.result_1>=results_df.result_2].result_1.values
loser_1  = results_df[results_df.result_1>=results_df.result_2].result_2.values

winner_2 = results_df[results_df.result_1<results_df.result_2].result_2.values
loser_2  = results_df[results_df.result_1<results_df.result_2].result_1.values

winner = np.concatenate((winner_1,winner_2))
loser = np.concatenate((loser_1,loser_2))
scores_df = pd.DataFrame(np.vstack((winner,loser)).T,columns=['winner','loser'])


# In[ ]:


gb = scores_df.groupby(by=['winner','loser'])['winner'].count()/scores_df.shape[0]
overtime_percentage = str(round(gb[gb.index.get_level_values(0)!=16].sum()*100,1))+'%'

gb = round(gb[gb>10**-3]*100,1)

index_plot = np.array(gb.index.get_level_values(0).astype('str'))+'-'+np.array(
    gb.index.get_level_values(1).astype('str'))

fig = go.Figure()
fig.add_trace(go.Scatter(x=index_plot,y=gb.values, name='results'))
fig.update_layout(xaxis_type='category',title='Scores distribution',xaxis_title='Score',yaxis_title='Percentage of matches (%)')


# In[ ]:


overtime_percentage


# We can see that on regular time (disregarding overtime), the most common score is 16-14 (achieved in 10.7% of the matches) and the rarest score is 16-0 (achieved in only 0.2% of the matches), with intermediate scores falling somewhere in between. 
# 
# 9.7% of the matches go to overtime.
# 
# The results may differ if we consider matches played by non top-tier teams.

# ## Most CT sided map

# There has long been a dispute in CS:GO to determine the most CT-sided maps, and always present discussion if having a heavily one-sided map is a desirable outcome. Here we determine this characteristic by computing the average scores obtained in each side of the map and then comparing both sides.

# In[ ]:


ct_1 = results_df[['date','_map','ct_1']].rename(columns={'ct_1':'ct'})
ct_2 = results_df[['date','_map','ct_2']].rename(columns={'ct_2':'ct'})
ct = pd.concat((ct_1,ct_2))


# In[ ]:


t_1 = results_df[['date','_map','t_1']].rename(columns={'t_1':'t'})
t_2 = results_df[['date','_map','t_2']].rename(columns={'t_2':'t'})
t = pd.concat((t_1,t_2))


# In[ ]:


t = t.sort_values('date')
ct = ct.sort_values('date')


# In[ ]:


maps = ['Cache','Cobblestone','Dust2','Inferno','Mirage','Nuke','Overpass','Train','Vertigo']


# In[ ]:


series_t, series_ct, how_ct = {},{},{}
for i, key in enumerate(maps):
    t_map = t[t._map == maps[i]]
    ct_map = ct[ct._map == maps[i]]
    y_t = t_map.t.rolling(min_periods = 20, window= 200, center=True).sum().values
    y_ct = ct_map.ct.rolling(min_periods = 20, window= 200, center=True).sum().values
    
    series_t[key] = pd.Series(data=y_t,index=t_map.date)
    series_ct[key] = pd.Series(data=y_ct,index=ct_map.date)
    
    how_ct[key] = series_ct[key]/(series_ct[key]+series_t[key])//0.001/10


# In[ ]:


def add_trace(_map):
    fig.add_trace(go.Scatter(x=how_ct[_map].index, y=how_ct[_map].values, name=_map))


# In[ ]:


fig = go.Figure()
for _map in maps:
    add_trace(_map)
fig.add_trace(go.Scatter(x=['2015-11-01', '2020-03-12'], y=[50,50],
                         mode='lines',line=dict(color='grey'),showlegend=False))
fig.update_layout(title='Distribution of rounds between CT and T sides',
                  yaxis_title='Percentage of round won on the CT-side (%)')
fig.show()


# There are long stretches without data for a map in the graph. This happens because maps are added and removed from the map pool constantly.
# 
# <b>Nuke</b> and <b>Train</b> oscilatte as being the most CT-sided maps, having around 57% of the rounds won on the CT-side, while
# <b>Dust2</b> and <b>Cache</b> are historically the most T-sided maps.
# 
# It is interesting to note that <b>Inferno</b> was known for being a heavily CT-sided map prior to 2016, which was one of the reasons to update it. Since its update, <b>Inferno</b> has actually been the most balanced map in this aspect.
# 
# 

# ## Maps played per period

# About the maps:
#  - <b>Mirage</b>, <b>Train</b>, <b>Inferno</b> and <b>Overpass</b> are the maps from which we have the most data available. They are also the maps present on the map pool for the longest time;
# 
#  - <b>Cache</b>, <b>Cobblestone</b> and <b>Dust2</b> have been played less, but have also been outside the map pool for the longest periods:
# 
#  - <b>Nuke</b> is historically the least played map, even though it has been present in the map pool for a long time. The only explanation for this stems from the teams' unfamiliarity with the map;
# 
#  - <b>Vertigo</b> has limited data available as it was the most recently added map in the map pool.

# In[ ]:


print('Total number of matches played on the map:')
results_df.groupby('_map').date.count()


# In CS:GO, the most reputable tournaments are the Majors. These tournaments are normally played twice a year and have a prize pool of $1,000,000. More information about the subject can be seen here: https://liquipedia.net/counterstrike/Majors
# 
# For the next step, we are going to discretize the 'date' column in a dataframe into a 'time_period' column. This new column will refer to the most recently played major tournament.
# 
# As an example, following this binning technique, we are currently on the Berlin 2019 period, as that was the most recently played tournament.

# In[ ]:


majors = [{'tournament':'01. Cluj-Napoca 2015','start_date':'2015-10-28'},
          {'tournament':'02. Columbus 2016','start_date':'2016-03-29'},
          {'tournament':'03. Cologne 2016','start_date':'2016-07-05'},
          {'tournament':'04. Atlanta 2017','start_date':'2017-01-22'},
          {'tournament':'05. Krakow 2017','start_date':'2017-07-16'},
          {'tournament':'06. Boston 2018','start_date':'2018-01-26'},
          {'tournament':'07. London 2018','start_date':'2018-09-20'},
          {'tournament':'08. Katowice 2019','start_date':'2019-02-28'},
          {'tournament':'09. Berlin 2019','start_date':'2019-09-05'}]


# In[ ]:


def create_col_time_period(df):
    df['time_period'] = ''
    
    for major_start in majors:
        df.loc[(df['date']>=major_start['start_date']),'time_period'] = major_start['tournament']
    
    return df


# In[ ]:


results_df = create_col_time_period(results_df)
economy_df = create_col_time_period(economy_df)
picks_df = create_col_time_period(picks_df)
players_df = players_df.merge(results_df[['match_id','time_period']],'left',on='match_id')


# In[ ]:


results_df_team_1 = results_df[['time_period','team_1','_map','ct_1','t_2','ct_2','t_1']
                      ].rename(columns={'team_1':'team'})
results_df_team_2 = results_df[['time_period','team_2','_map','ct_1','t_2','ct_2','t_1']
                      ].rename(columns={'team_2':'team'})
results_df_teams = pd.concat((results_df_team_1,results_df_team_2))[['time_period','team','_map']]


# In[ ]:


gb = results_df_teams.groupby(['time_period','_map']).team.count()
gb_text = round(gb*100/gb.groupby('time_period').sum(),1).reset_index().rename(columns={'team':'percentage'})
gb_text.percentage = gb_text.percentage.astype(str)+'%'
gb = gb.reset_index()


# In[ ]:


fig = go.Figure()
for _map in maps:
    fig.add_bar(name=_map,x=gb[gb._map==_map].time_period,y=gb[gb._map==_map].team,
                text=gb_text[gb_text._map==_map].percentage,textposition='inside')

fig.update_layout(barmode='stack',legend=dict(traceorder='normal'),yaxis_title='Number of maps played',font=dict(size=10))
fig.show()


# As pointed out previously, <b>Nuke</b> is historically the least popular map in the pool. This has been changing recently, as teams that used to permaban the map have moved on to banning maps like Vertigo.
# 
# <b>Vertigo</b>, as the newest and most unconventional map, is also the most unpopular map, probably due to the many changes it has had in its short competitive term.
# 
# The period between Columbus and Cologne 2016 has the least amount of maps played and is also the shortest (under 4 months), while the period between Boston and London 2018 has the highest amount of maps played and is also the longest (over 7 months).

# ## Best teams on each map

# In this section, let's search for the best teams in each map. The victory percentage for the teams on each map is readily available in many e-sports sites like hltv.org, so it would be pointless to present the information the same way. For this reason, instead of pointing out the percentages, we are going to rank the best teams on each map and each side (CT and T) for every time period.

# In[ ]:


results_df_team_1_ct = results_df_team_1.rename(columns={'ct_1':'ct_team','t_2':'t_opponent'}).drop(columns=['ct_2','t_1'])
results_df_team_2_ct = results_df_team_2.rename(columns={'ct_2':'ct_team','t_1':'t_opponent'}).drop(columns=['ct_1','t_2'])
results_df_ct = pd.concat((results_df_team_1_ct,results_df_team_2_ct),sort=True)

results_df_team_1_t = results_df_team_1.rename(columns={'t_1':'t_team','ct_2':'ct_opponent'}).drop(columns=['ct_1','t_2'])
results_df_team_2_t = results_df_team_2.rename(columns={'t_2':'t_team','ct_1':'ct_opponent'}).drop(columns=['ct_2','t_1'])
results_df_t = pd.concat((results_df_team_1_t,results_df_team_2_t),sort=True)


# In[ ]:


results_df_ct['side_diff'] = results_df_ct['ct_team']-results_df_ct['t_opponent']
results_df_ct['side_sum'] = results_df_ct['ct_team']+results_df_ct['t_opponent']

results_df_t['side_diff'] = results_df_t['t_team']-results_df_t['ct_opponent']
results_df_t['side_sum']  = results_df_t['t_team'] +results_df_t['ct_opponent']

results_df_ct.head()


# In[ ]:


def groupby_time_map_team(results_df_side):
    gb = results_df_side.groupby(['time_period','_map','team'])['side_diff','side_sum'].sum()
    gb['side_diff_per_game'] = gb['side_diff']/(gb['side_sum']/15)
    gb = gb.sort_values(['time_period','_map','side_diff_per_game'],ascending=[1,1,0])

    for major in majors:
        col = major['tournament']
        _filter = (gb.side_sum > gb.loc[col].side_sum.mean()*3/4)
        gb.loc[col] = gb.loc[_filter][gb.loc[_filter].index.get_level_values(0)==col]

    gb.dropna(inplace=True)    

    return gb


# In[ ]:


gb_ct = groupby_time_map_team(results_df_ct)
gb_t = groupby_time_map_team(results_df_t)


# In[ ]:


def plot_ranking_teams_sides(gb):
    rankings_teams = {}
    for _map in maps:
        rankings_teams[_map] = pd.DataFrame(index=range(1,6),)
        rankings_teams[_map].index.name = 'ranking'
        rankings_teams[_map].style.set_caption(_map)

        for major in majors:
            col = major['tournament']
            try:
                rankings_teams[_map][col] = gb.loc[col,_map]['side_diff_per_game'][:5].index
            except:
                pass
        print('\n'+_map+':')
        display(rankings_teams[_map])


# In[ ]:


print('T-side Rankings:\n')
plot_ranking_teams_sides(gb_t)


# In[ ]:


print('CT-side Rankings:\n')
plot_ranking_teams_sides(gb_ct)


# # Economy DataFrame

# ## Round victory probability by equipment value

# Counter-strike has an economic system that governs the acquisitions of armor, weapons and grenades by the players. The rules of this system have changed many times in the past. 
# 
# Currently, round loss bonuses are based on a count of a team's round loss bonus. This count is increased by one after every loss and decreased by one after every win (minimum 0). The money returned after losing a round is:
# 
#  - 0 Losses: &dollar;1400;
#  - 1 Loss: &dollar;1900;
#  - 2 Losses: &dollar;2400;
#  - 3 Losses: &dollar;2900;
#  - 4+ Losses: &dollar;3400.
# 
# Previously to 2019, a win would reset the round loss bonus back to &dollar;1400 per player. Now, a win reduces the loss count by one.
# Teams begin the half with a loss count of 1, so that losing the pistol round is not as damaging to the economy.
# 
# Players start the half with &dollar;800, which is enough to buy either armour or an improved pistol, but not both, nor better weapons. That is why rounds 1 and 16 are called pistol rounds.
# 
# Winning a round by eliminations grants &dollar;3250 to the winning team. If the win was achieved by defusing the bomb as CT or by having it detonate as T, the reward is of 3500 dollars.
# 
# Another way of getting money in the match is by killing players in the opposing team. The money received per kill varies according to the weapon wielded:
# - &dollar;100: Sniper rifle kills;
# - &dollar;300: pistol, assault rifle and grenade kills;
# - &dollar;600: SMG kills;
# - &dollar;900: Shotgun kills;
# - &dollar;1500: Knife kills.

# In the HLTV economy section of the matches, the sum of the equipment value of the teams is categorized according to 4 equipment value ranges:
#  - &dollar; 0-5 K: Eco;
#  - &dollar; 5-10 K: Forced Pistols;
#  - &dollar; 10-20 K: Semi-buy;
#  - &dollar; 20+ K: Full buy.
#  
# In this notebooks, the Semi-buy category was split into two new categories, according to equipment value:
#  - &dollar;10-15 K: ForcedSMGs;
#  - &dollar;15-20 K: ForcedBuy.
#  
# This change was added because the &dollar;10-20 K money range is too broad and groups together very distinct sets of equipment.

# In[ ]:


economy_df.head()


# In[ ]:


money_columns = ['2_t1','3_t1','4_t1','5_t1','6_t1','7_t1','8_t1','9_t1','10_t1','11_t1','12_t1','13_t1','14_t1'
                ,'15_t1','17_t1','18_t1','19_t1','20_t1','21_t1','22_t1','23_t1','24_t1','25_t1','26_t1','27_t1',
                 '28_t1','29_t1','30_t1',
                '2_t2','3_t2','4_t2','5_t2','6_t2','7_t2','8_t2','9_t2','10_t2','11_t2','12_t2','13_t2','14_t2'
                ,'15_t2','17_t2','18_t2','19_t2','20_t2','21_t2','22_t2','23_t2','24_t2','25_t2','26_t2','27_t2',
                 '28_t2','29_t2','30_t2']

economy_categories = {0:{'name':'eco','start':0,'end':5000},
                      1:{'name':'forcedPistols','start':5000,'end':10000},
                      2:{'name':'forcedSMGs','start':10000,'end':15000},
                      3:{'name':'forcedBuy','start':15000,'end':20000},
                      4:{'name':'fullBuy','start':20000,'end':50000}
                      }


# In[ ]:


for col in money_columns:
    for key, category in economy_categories.items():
        economy_df.loc[(economy_df[col]>category['start']) & (economy_df[col]<=category['end']),col] = key
    for key, category in economy_categories.items():
        economy_df.loc[economy_df[col]==key,col] = category['name']


# In[ ]:


def get_economy_stats(category):

    wins_by_side_t1 = pd.DataFrame([[0,0,0],[0,0,0]],index=['ct','t'],columns=['sum','count','mean'])
    wins_by_side_t2 = pd.DataFrame([[0,0,0],[0,0,0]],index=['ct','t'],columns=['sum','count','mean'])

    for _round in range(2,16):
        gb_1 = economy_df[economy_df[str(_round)+'_t1']==category].rename(columns={'t1_start':'side'}).groupby('side')[str(_round)+'_winner']
        gb_1 = gb_1.agg(['sum','count','mean'])

        gb_3 = economy_df[economy_df[str(_round+15)+'_t1']==category].rename(columns={'t2_start':'side'}).groupby('side')[str(_round+15)+'_winner']
        gb_3 = gb_3.agg(['sum','count','mean'])

        gb_1 = gb_1.reindex(['ct','t'], fill_value=0)
        gb_3 = gb_3.reindex(['ct','t'], fill_value=0)

        wins_by_side_t1 = wins_by_side_t1 + gb_1 + gb_3

    wins_by_side_t1['sum'] = 2*wins_by_side_t1['count']-wins_by_side_t1['sum']

    for _round in range(2,16):
        gb_2 = economy_df[economy_df[str(_round)+'_t2']==category].rename(columns={'t2_start':'side'}).groupby('side')[str(_round)+'_winner']
        gb_2 = gb_2.agg(['sum','count','mean'])

        gb_4 = economy_df[economy_df[str(_round+15)+'_t2']==category].rename(columns={'t1_start':'side'}).groupby('side')[str(_round+15)+'_winner']
        gb_4 = gb_4.agg(['sum','count','mean'])

        gb_2 = gb_2.reindex(['ct','t'], fill_value=0)
        gb_4 = gb_4.reindex(['ct','t'], fill_value=0)

        wins_by_side_t2 = wins_by_side_t2 + gb_2 + gb_4

    wins_by_side_t2['sum'] = wins_by_side_t2['sum']-wins_by_side_t2['count']

    wins_by_side = wins_by_side_t1 + wins_by_side_t2

    wins_by_side['mean'] = wins_by_side['sum']/wins_by_side['count']//0.001/10
    wins_by_side['num_per_game'] = wins_by_side['count']/economy_df.shape[0]//0.1/10
    wins_by_side = wins_by_side[['mean','num_per_game']]
    
    return wins_by_side


# In[ ]:


economy_stats = {}
mean_victories_df = pd.DataFrame(index=['ct','t'])
num_per_game_df = pd.DataFrame(index=['ct','t'])

for category in economy_categories.values():
    cat = category['name']
    economy_stats[cat] = get_economy_stats(cat)
    mean_victories_df[cat] = economy_stats[cat]['mean']
    num_per_game_df[cat] = economy_stats[cat]['num_per_game']

print('\nVictory probability (%):')
display(mean_victories_df)
print('\nNumber per game:')
display(num_per_game_df)


# The tables above present the probability of winning rounds according to the team's equipment value category, and the number of rounds of each category type in a map. The results do not take into consideration the equipment category of the opposing team.
# 
# Teams are 40% more likely to win <b>eco</b> rounds on the CT-side, 10% more likely to win <b>forcedBuy</b> rounds on the T-side and 11% more likely to win <b>fullBuy</b> rounds on the CT-side.
# 
# A team has on average 6 <b>non-fullBuy</b> rounds in a map. The number of rounds in each category does not vary much between CT and T sides, which is surprising, as the CT-side economy usually seems to be the hardest one to maintain. 

# ## Pistols ranking

# In[ ]:


gb_team_1_first_pistol   = economy_df.rename(columns={'team_1':'team','t1_start':'side'}).groupby(['side','time_period','team'])['1_winner'].agg(['mean','count'])
gb_team_1_second_pistol  = economy_df.rename(columns={'team_1':'team','t2_start':'side'}).groupby(['side','time_period','team'])['16_winner'].agg(['mean','count'])

gb_team_2_first_pistol   = economy_df.rename(columns={'team_2':'team','t2_start':'side'}).groupby(['side','time_period','team'])['1_winner'].agg(['mean','count'])
gb_team_2_second_pistol  = economy_df.rename(columns={'team_2':'team','t1_start':'side'}).groupby(['side','time_period','team'])['16_winner'].agg(['mean','count'])


# In[ ]:


gb = (2-gb_team_1_first_pistol['mean'])*gb_team_1_first_pistol['count']+(
    2-gb_team_1_second_pistol['mean'])*gb_team_1_second_pistol['count']+(
    gb_team_2_first_pistol['mean']-1)*gb_team_2_first_pistol['count']+(
    gb_team_2_second_pistol['mean']-1)*gb_team_2_second_pistol['count']

total_pistols = (gb_team_1_first_pistol['count']+gb_team_1_second_pistol['count']+gb_team_2_first_pistol['count']+gb_team_2_second_pistol['count'])

for major in majors[3:]:
    col = major['tournament']
    
    _filter = total_pistols > total_pistols.loc[:,col].quantile(0.3)
    
    gb.loc[:,col] = gb.loc[_filter,col]
    total_pistols.loc[:,col] = total_pistols.loc[_filter,col]
    
    gb.dropna(inplace=True)
    total_pistols.dropna(inplace=True)

mean_pistols = pd.DataFrame(gb/total_pistols)
mean_pistols.dropna(inplace=True)
mean_pistols.sort_values(['side','time_period',0],ascending=[1,1,0],inplace=True)


# In[ ]:


def get_rankings_pistols_side(side):
    ranking_pistols_side = pd.DataFrame(index=range(1,8))
    ranking_pistols_side.index.name = 'ranking'

    for major in majors[3:]:
        col = major['tournament']
        ranking_pistols_side[col] = mean_pistols.loc[side,col][0][:7].index
    
    return ranking_pistols_side


# In[ ]:


print('\nRankings Pistols CT-side:')
display(get_rankings_pistols_side('ct'))
print('\nRankings Pistols T-side:')
display(get_rankings_pistols_side('t'))


# # Players DataFrame

# ## Players ranking by map

# In[ ]:


players_df.head()


# In[ ]:


all_maps_columns = ['date','time_period','country','player_name','team','opponent','player_id',
                    'match_id','event_id','event_name','best_of']
each_map_columns = ['kills','assists','deaths','hs','flash_assists','kast','kddiff','adr','fkdiff','rating']


# In[ ]:


map1_columns = ['map_1']+['m1_'+ x for x in each_map_columns]
map2_columns = ['map_2']+['m2_'+ x for x in each_map_columns]
map3_columns = ['map_3']+['m3_'+ x for x in each_map_columns]


# In[ ]:


out_columns = all_maps_columns+['_map']+each_map_columns

players_df_by_map_columns = pd.DataFrame(columns=out_columns)


# In[ ]:


#Countries that contribute the most to the professional scene by number of matches
players_df.groupby('country')['country'].count().sort_values(ascending=False)[:30]


# In[ ]:


curr_map = {}
curr_map[0] = players_df[(all_maps_columns+map1_columns)]
curr_map[1] = players_df[(all_maps_columns+map2_columns)]
curr_map[2] = players_df[(all_maps_columns+map3_columns)]

curr_map[0].columns = out_columns
curr_map[1].columns = out_columns
curr_map[2].columns = out_columns

all_maps = pd.concat(   (   pd.concat(   (curr_map[0],curr_map[1])    ), curr_map[2]   )   )


# In[ ]:


gb2 = all_maps.groupby(['time_period','player_id','_map'])
threshold_maps_played = 7
all_maps2 = gb2.filter(lambda x:x.player_name.count()>threshold_maps_played)
all_maps2.head()


# In[ ]:


gb = all_maps2.groupby(['time_period','_map','player_name'],sort=False)['rating','kddiff'].mean()
rankings = gb.sort_values(['time_period','_map','rating'],ascending=[1,1,0])


# In[ ]:


rankings_players = {}
for _map in maps:
    rankings_players[_map] = pd.DataFrame(index=range(1,21))
    rankings_players[_map].index.name = 'ranking'
    
    for major in majors:
        col = major['tournament']
        try:
            rankings_players[_map][col] = rankings.loc[col,_map].rating[:20].index
        except:
            pass


# Best players on each map accordding to their average Rating 2.0:

# In[ ]:


for _map in maps:
    print('\n'+_map+':')
    display(rankings_players[_map])


# ## Players ranking all maps

# In[ ]:


ranking_players_df = pd.DataFrame()
ranking_players_df['player'] = players_df.player_name.unique()
ranking_players_df.set_index('player',inplace=True)

for major in majors:
    col = major['tournament']
    ranking_players_df[col] = 0
    
for _map in maps:
    for col in rankings_players[_map].columns:
        for i in range(1,21):
            ranking_players_df.loc[rankings_players[_map][col][i],col] += 21-i
            
rankings_players_again = {}

rankings_players_again = pd.DataFrame(index=range(1,21))
rankings_players_again.index.name = 'ranking'

for major in majors:
    col = major['tournament']
    rankings_players_again[col] = ranking_players_df[col].sort_values(ascending=False)[:20].index


# Below it is shown the player rankings if it was only determined by the rating 2.0 across all maps. It can be considered as the ranking of most versatile fraggers. 

# In[ ]:


rankings_players_again


# ## K/D Graph

# In[ ]:


results_df_rank_part_1 = results_df[['match_id','team_1','rank_1']].rename(columns={'team_1':'team','rank_1':'team_rank'})
results_df_rank_part_2 = results_df[['match_id','team_2','rank_2']].rename(columns={'team_2':'team','rank_2':'team_rank'})
results_df_rank = pd.concat((results_df_rank_part_1,results_df_rank_part_2))


# In[ ]:


all_maps3 = all_maps2.merge(results_df_rank,'left',on=['match_id','team'])


# In[ ]:


players_series = all_maps3.groupby('player_name').team_rank.min()
players_list = list(players_series[players_series<=3].index)


# In[ ]:


gb = all_maps3.groupby(['time_period','player_name','country'])['kills','deaths','team_rank']
gb = gb.mean()[gb.count()['kills']>100]
gb = gb[gb.index.get_level_values(1).isin(players_list)]

gb['kills'] = gb['kills'].round(1)
gb['deaths'] = gb['deaths'].round(1)
gb['team_rank'] = gb['team_rank'].round(0).astype('int')
gb.reset_index(inplace=True)


# In[ ]:


gb['region'] = ''
gb.loc[(gb['country']=='Ukraine') | (gb['country']=='Russia') | (gb['country']=='Kazakhstan'),'region'] = 'CIS'
gb.loc[(gb['country']=='Brazil'),'region'] = 'Brazil'
gb.loc[(gb['country']=='France') | (gb['country']=='Belgium'),'region'] = 'France/Belgium'
gb.loc[(gb['country']=='United States') | (gb['country']=='Canada'),'region'] = 'North America'
gb.loc[(gb['country']=='Denmark'),'region'] = 'Denmark'
gb.loc[(gb['country']=='Sweden'),'region'] = 'Sweden'
gb.loc[(gb['country']=='Poland'),'region'] = 'Poland'
gb.loc[gb.country.isin(['Netherlands','Slovakia','Bosnia and Herzegovina',
                        'Norway','Czech Republic','Spain','Estonia','United Kingdom','Portugal','Turkey','Bulgaria', 'Finland']),'region'] = 'Rest of Europe'

gb = gb.sort_values(['time_period','region'])


# In[ ]:


gb['size'] = (100/(gb['team_rank']+2)).round(1)


# In[ ]:


fig = px.scatter(gb, x="deaths", y="kills", animation_frame="time_period", animation_group="player_name",
           size="size", color="region", hover_name="player_name", hover_data=["team_rank"],
                 range_x=[14,24],range_y=[14,24]
                )
#.for_each_trace(lambda t: t.update(name=t.name.split("=")[1]))
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 1300
fig.update_layout(xaxis_title='Deaths', yaxis_title = 'Kills')
fig.add_shape(type="line", x0=14, y0=14, x1=24, y1=24, line=dict(width=4, dash="dot"))
fig.update_shapes(dict(xref='x', yref='y'))
fig.show()


# # Picks DataFrame

# In[ ]:


picks_df.head()


# In[ ]:


gb = picks_df.groupby('system').system.count()
gb = gb[gb>10]
gb


# The order 123412 (t1_remove, t2_remove, t1_pick, t2_pick, t1_remove, t2_remove, left_over) is by far the most common map picking system for bo3 matches.
# For bo1 matches, there is a three-way split between the systems 122112, 112221 and 121212.

# # Further exploration and predictions

# If you think these insights are useful, I might update the dataset with data from 2014 and 2015.
# 
# What insights can you create from this data? Can you predict the probability of each:
#  - team winning a map?
#  - map being vetoed or selected?
#  - team winning the match, combining predictions from map winners and map picks?
#  
#  Give it a try!
# 
