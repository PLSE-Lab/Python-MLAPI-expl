#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Setting package umum 
import pandas as pd
import pandas_profiling as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pylab import rcParams
# For every plotting cell use this
# grid = gridspec.GridSpec(n_row,n_col)
# ax = plt.subplot(grid[i])
# fig, axes = plt.subplots()
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.options.display.float_format = '{:.5f}'.format

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Load dataset
df = pd.read_csv('/kaggle/input/josh-devlin-2018-nba-playoff-shoot/playoff_shots.csv')


# # Shot to Championness
# Two years have past from the 2018 NBA Playoff where the Golden State Warriors, arguably the greatest shooting team in NBA history crowned champion, sweeping the Cleveland Cavaliers 4-0. This remarks the end of the GSW vs Cleveland rivalries because now in 2020 neither team is as good as it's prime. Well actually GSW still made the final next year but eventually lose to Toronto Raptors. When I stumbled upon these dataset made by Josh Devlin that consist of all the shooting attempt in the NBA Playfoss 2018 game I can't help myself to be curious about how goood GSW  actually in shooting. 
# 
# Inspired by : Josh Devlin article about streaks analysis (https://joshdevlin.com/blog/calculate-streaks-in-pandas/?utm_campaign=Data_Elixir&utm_source=Data_Elixir_285)

# In[ ]:


# Dataset overview
df.head(10)


# The dataset differ the type of shot by three categories : 3PT, 2PT, and FT. For each row it tells what is the type of the shot attempted, the exact time, the player who attempt it, and in which game.

# In[ ]:


# Get all the player on each team
dict_player = {}
for team in df['team'].unique() :
    dict_player[team] = df[df['team']==team]['player_name'].unique()
    
display(dict_player['Golden State Warriors'])


# In[ ]:


### Function to generate shooting summary
def shooting_summary(df, by) :
    
    # Filter dataset
    df_summary = df.copy()
    df_3p = df_summary[df_summary['shot_type']=='3PT']
    df_2p = df_summary[df_summary['shot_type']=='2PT']
    df_ft = df_summary[df_summary['shot_type']=='FT']
    result = pd.DataFrame()
    
    # Count FGA  and FG% for all attempt
    all_make = df_summary[df_summary['result']=='make'].groupby(by).count()['game_id']
    all_miss = df_summary[df_summary['result']=='miss'].groupby(by).count()['game_id']
    result['Shot Attempts'] = all_make + all_miss
    result['Shot %'] = all_make / result['Shot Attempts'] * 100
    
    # Count FGA and FG% for 3p
    make_3p = df_3p[df_3p['result']=='make'].groupby(by).count()['game_id']
    miss_3p = df_3p[df_3p['result']=='miss'].groupby(by).count()['game_id']
    result['3p Attempts'] = make_3p + miss_3p
    result['3p %'] = make_3p / result['3p Attempts'] * 100
    
    # Count FGA and FG% for 2p
    make_2p = df_2p[df_2p['result']=='make'].groupby(by).count()['game_id']
    miss_2p = df_2p[df_2p['result']=='miss'].groupby(by).count()['game_id']
    result['2p Attempts'] = make_2p + miss_2p
    result['2p %'] = make_2p / result['2p Attempts'] * 100
    
    # Count FGA and FG% for FT
    make_ft = df_ft[df_ft['result']=='make'].groupby(by).count()['game_id']
    miss_ft = df_ft[df_ft['result']=='miss'].groupby(by).count()['game_id']
    result['FT Attempts'] = make_ft + miss_ft
    result['FT %'] = make_ft / result['FT Attempts'] * 100
    
    # Calculate proportion of team shot attempts
    result['TS Proportion'] = result['Shot Attempts'] / np.sum(result['Shot Attempts']) * 100
    
    return result


# So without further ado let's see how GSW player shot their shot. For now let just see the top 5 player based by the total shot attempts

# In[ ]:


### Shooting summary of GSW
rcParams['figure.figsize'] = [15,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
grid = gridspec.GridSpec(1,2)

# Plot prep
gsw = df[df['team']=='Golden State Warriors']
df_plot = shooting_summary(gsw, 'player_name').fillna(0).sort_values('Shot Attempts', ascending=False).head(5)

# Plot shooting percentage
ax1 = plt.subplot(grid[0])
sns.heatmap(df_plot[['Shot %', '3p %', '2p %', 'FT %', 'TS Proportion']], annot=True, ax=ax1, cmap='Pastel1_r', linewidths=2)
ax1.set_title('GSW Shooting Percentage', fontname='Monospace', fontsize='20', fontweight='bold')

# Plot shooting attempts
ax2 = plt.subplot(grid[1])
sns.heatmap(df_plot[['3p Attempts', '2p Attempts', 'FT Attempts']], annot=True, fmt='.2f', cmap='Pastel1_r', ax=ax2, linewidths=2)
ax2.set_title('GSW Shooting Attempts', fontname='Monospace', fontsize='20', fontweight='bold')
plt.tight_layout()


# We can see that Kevin Durant (you know who he is) carry 26.27% of all GSW shooting attempts. That is a lot of shot and thankfully he made most of them, shooting 34.05% from three (good but not great number) and a preposterous 55.55% from two on 297 attempts. This guy have a monster mid game. On top of that he get 152 FT Attempts (WTF !?) and made 90.13% of them. These number indeed worthy of NBA Finals MVP. 
# 
# Aside from that we can see that Klay Thompson and Stephen Curry (the splash bro) give significant impact specifically in 3p, shooting 42.67% and 39.50% in more attempts than Kevin Durant. No wonder GSW win it all because they are indeed eltie in shooting.
# 
# Kevin Durant indeed shot a big proportion of the team attempt, but are there a player that take a lot their team shot more than him ?

# In[ ]:


### Check wether a player like to shoot more than KD
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')

# Plot prep
kevin_durant_prop = 26.27570
result = pd.DataFrame()

for team in df['team'].unique() :
    df_team = df[df['team']==team]
    summary = shooting_summary(df_team, 'player_name')
    
    bool_team_shot = summary['TS Proportion'] >= kevin_durant_prop
    result = pd.concat([result, summary[bool_team_shot]])
    
# Plot
cmap = sns.color_palette("Pastel1_r")
result['TS Proportion'].sort_values().plot(kind='barh', color=cmap)

# Add cosmetics
list_x = list(np.round(result['TS Proportion'].sort_values(), 2))
list_y = list(np.arange(-0.1, 5.9, 1)) 
for x,y in zip(list_x,list_y) :
    plt.text(x + 0.5, y, str(x), fontweight='bold', fontsize=14, fontname='Monospace')
    
plt.plot([0, np.max(list_x)], [0.5, 0.5], '--', color='grey')
plt.text(np.max(list_x) + 0.5, 0.4, 'Threshold', fontname='Monospace', fontsize=14)
    
plt.xlabel('Team Shot Proportion')
plt.title('Top Player based on TS Proportion', fontname='Monospace', fontsize=20, fontweight='bold')
plt.xlim(0, np.max(list_x) * 1.2) ;


# So there are 5 player with a bigger Team Shot Proportion than Kevin Durant, all of them is a noticeable player, 3 NBA MVP Finalist on LeBron James (King), Anthony Davis, and the reigning MVP itself James Harden. The remaining two is Russel Westbrook (Last year MVP) and John Wall. LeBron James itself made the NBA Finals match again GSW, so this number really represent how LeBron is very valuable for his team (and how bad the team will be without him). Lets compare LeBron production with Kevin Durant

# In[ ]:


### Compare KD and LeBron
def highlight_max(s):
    '''
    highlight the maximum 
    '''
    is_max = s == s.max()
    return ['background-color:#f85a40' if v else '' for v in is_max]

print('--- Comparison of Kevin Durant and Lebron James throughout the playoffs 2018 ---')
display(result[result.index.isin(['LeBron James','Kevin Durant'])].T.style.apply(highlight_max, axis=1))


# Clearly we can see that LeBron James lead in almost all of the shooting statistics, except in FT% where he shoot an okay 74.64%. He even dominate the mid and 3p shot better than KD and he is not famous for his shooting. This number truly represent how unbelieveable LeBron James is. But eventually he got sweeped in the final by GSW. So lets see how the shooting in the finals occur by both player

# In[ ]:


### Shooting stats in the Finals only
df_final = df[df['game_description'].str.contains('CLE v GS')]
result = shooting_summary(df_final, 'player_name')

print('--- Comparison of Kevin Durant and Lebron James in NBA Finals 2018 ---')
display(result[result.index.isin(['LeBron James','Kevin Durant'])].T.style.apply(highlight_max, axis=1))


# Clearly from here we can see that the statistics have turn to Kevin Durant favor. He shot more efficiently in all of the type of shot than LeBron James. The only thing LeBron do better is taking more shot (lot of carrying to do).
# 
# Well enough of this comparison lets see how truly well GSW as team shot their shot compare to other team. We will calculate the statistics for each round of the playoff and rank them. To really see how consistent GSW shot to championness

# In[ ]:


### Grand comparison
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')

list_var = ['Shot %','3p %','2p %','FT %']
team = 'Golden State Warriors'
result = pd.DataFrame()

# First round
first_round_desc = list(df['game_description'].unique())[:44]
first_round_desc.append('CLE v IND, G7')
first_round = df[df['game_description'].isin(first_round_desc)]
summary_first_round = shooting_summary(first_round, 'team')[list_var]
for var in list_var :
    summary_first_round[var] = summary_first_round[var].rank(ascending=False)
team_summary = summary_first_round[summary_first_round.index == team]
team_summary.index = ['First Round (16)']
result = pd.concat([result, team_summary])

# Conference Seminfinals
conf_semi_desc = list(df['game_description'].unique())[44:64]
conf_semi_desc.remove('CLE v IND, G7')
conf_semi = df[df['game_description'].isin(conf_semi_desc)]
summary_conf_semi = shooting_summary(conf_semi, 'team')[list_var]
for var in list_var :
    summary_conf_semi[var] = summary_conf_semi[var].rank(ascending=False, method='min')
team_summary = summary_conf_semi[summary_conf_semi.index == team]
team_summary.index = ['Conf. Semifinals (8)']
result = pd.concat([result, team_summary])

# Conference Finals
conf_finals_desc = list(df['game_description'].unique())[64:78]
conf_finals = df[df['game_description'].isin(conf_finals_desc)]
summary_conf_finals = shooting_summary(conf_finals, 'team')[list_var]
for var in list_var :
    summary_conf_finals[var] = summary_conf_finals[var].rank(ascending=False)
team_summary = summary_conf_finals[summary_conf_finals.index == team]
team_summary.index = ['Conf. Finals (4)']
result = pd.concat([result, team_summary])

# Finals
finals_desc = list(df['game_description'].unique())[78:]
finals = df[df['game_description'].isin(finals_desc)]
summary_finals = shooting_summary(finals, 'team')[list_var]
for var in list_var :
    summary_finals[var] = summary_finals[var].rank(ascending=False)
team_summary = summary_finals[summary_finals.index == team]
team_summary.index = ['Finals (2)']
result = pd.concat([result, team_summary])
                      
# Plot
sns.heatmap(result, annot=True, cmap='Pastel1', linewidths=2)
plt.title('GSW Shooting % Rank', fontname='Monospace', fontsize='20', fontweight='bold') ;


# We can see that throughout the playoff on each round GSW consistently rank at the top 3 in shooting efficiency with a perfect rank 1 in all type of shot in Conference Final and the Finals. But surprisingly it is not because of their 3p shooting but instead their 2p and FT efficiency. GSW rank 13 and 7 in the First Round and Conference Semifinals on 3p %. It seems like their 3p exponentially getting better throughout the playoff. Indeed I remember that GSW in 2018 is known to explode late in the game, especially in the third quarter. So lets check if its true or not

# In[ ]:


### 3p of GSW in 3rd period
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
grid = gridspec.GridSpec(1,1)

# Prep plot
df_plot = shooting_summary(gsw, 'period').iloc[:4].reset_index()
df_plot['period'] = df_plot['period'].astype('str')

# Plot
ax1 = plt.subplot(grid[0])
sns.barplot(x=df_plot['period'], y=df_plot['3p Attempts'], ax=ax1, color='#4298b5', alpha=0.5)
ax1.set_ylim([0,180])

ax2 = ax1.twinx()
sns.lineplot(x=df_plot['period'], y=df_plot['3p %'], color='#f85a40', ax=ax2)
ax2.set_ylim([25, 46]) ;

# Add cosmetics
plt.title('3p of GSW based on period', fontname='Monospace', fontsize='20', fontweight='bold') ;


# We can see that it is true that GSW are a beast at 3rd quarter. They shot at around 44% from three in the 3rd quarter, around 10% better than the 2nd and 4th quarter in similar total attempt. I can guess that it must not be because Steve Kerr is a good coach alone, it has to have a certain player that can spark this explosion. So next we will see the production number of GSW player in 3rd quarter compare to other quarter

# In[ ]:


### Compare production of each player based on period
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')

# Plot prep
list_player = ['Kevin Durant','Stephen Curry','Klay Thompson']
list_period = [1,2,3,4]

gsw = df[df['team']=='Golden State Warriors']
gsw_summary = shooting_summary(gsw, ['player_name','period']).reset_index()
gsw_summary = gsw_summary[gsw_summary['player_name'].isin(list_player)]
gsw_summary = gsw_summary[gsw_summary['period'].isin(list_period)]

# Plot
sns.barplot(data=gsw_summary, x='period',y='3p %', hue='player_name')

# Add cosmetics
plt.title('3p% of Top GSW player based on period', fontname='Monospace', fontsize='20', fontweight='bold') ;


# Turn out the beast is Klay Thompson, shooting a ridiculous 56% from three in the 3rd quarter. This is the thing that you can expect from a player that broke a record for most point in a quarter and guess in what quarter he did it, 3rd QUARTER ! (You guys should check it on youtube). Kevin Durant highest 3p % also come in the 3rd quarter, and both player % then shrink in the 4th quarter. Luckily they have Stephen Curry to the rescue with a solid 3p % of 45% in the 4th quarter
# 
# These three guys are quite amazing and for bonus here are the graph that represent their streaks of 3p shooting throughout the playoffs 2018

# In[ ]:


# Ordered the dataframe
sort_order = ["player_id", "shot_type", "game_id", "period", "period_time"]
ascending = [True, True, True, True, False]

df_ordered = df.sort_values(sort_order, ascending=ascending).reset_index(drop=True)


# In[ ]:


### Function for making streaks count
def making_streaks(df) :
    '''
    Need to have `result`, 'player_id', and 'shot_type' column in the dataframe
    The dataframe must be ordered
    '''
    
    # Shift the result
    streaks_df = df.copy().reset_index(drop=True)
    
    # Make `Streak Flag`
    check_result = (streaks_df['result'] != streaks_df['result'].shift(1))
    check_player = (streaks_df['player_id'] == streaks_df['player_id'].shift(1, fill_value=streaks_df.iloc[0]['player_id']))
    check_shot = (streaks_df['shot_type'] == streaks_df['shot_type'].shift(1, fill_value=streaks_df.iloc[0]['shot_type']))
    streaks_df['Streak Flag'] = check_result == (check_player & check_shot)
    
    # Make `End Streak Flag`
    check_result = (streaks_df['result'] != streaks_df['result'].shift(-1))
    check_player = (streaks_df['player_id'] == streaks_df['player_id'].shift(-1, fill_value=streaks_df.iloc[-1]['player_id']))
    check_shot = (streaks_df['shot_type'] == streaks_df['shot_type'].shift(-1, fill_value=streaks_df.iloc[-1]['shot_type']))
    streaks_df['End Streak Flag'] = check_result == (check_player & check_shot)
    
    # Make an ID for each streak
    streaks_df['Streak ID'] = streaks_df['Streak Flag'].cumsum()
    
    # Count how many shot in each streak
    streaks_df['Streak Count'] = streaks_df.groupby('Streak ID').cumcount() + 1
    
    # Make start streaks description
    start_target_cols = ['Start Game', 'Start Period', 'Start Period Time']
    source_cols = ["game_description", "period", "period_time"]
    for target,source in zip(start_target_cols, source_cols) :
        streaks_df.loc[streaks_df['Streak Flag'], target] = streaks_df.loc[streaks_df['Streak Flag'], source]
        streaks_df[target] = streaks_df[target].fillna(method='ffill')
        
    # Make end streaks description
    end_target_cols = ['End Game', 'End Period', 'End Period Time']
    source_cols = ["game_description", "period", "period_time"]
    for target,source in zip(end_target_cols, source_cols) :
        streaks_df.loc[streaks_df['End Streak Flag'], target] = streaks_df.loc[streaks_df['End Streak Flag'], source]
        
    list_col = ['player_name','team','shot_type','result', 'Streak Count', 'Streak ID'] + start_target_cols + end_target_cols
    
    return streaks_df, list_col


# In[ ]:


# Make summary of streaks for each player each shots
df_streaks, list_col = making_streaks(df_ordered)


# In[ ]:


# Plotting shoot progress
rcParams['figure.figsize'] = [15,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
fig, ax = plt.subplots()

# Prep plot
bool_shot = (df_streaks['shot_type'] == '3PT')
bool_player = (df_streaks['player_name'] == 'Kevin Durant')
df_plot_make = df_streaks[(bool_shot & bool_player)].reset_index()
bool_make = (df_plot_make['result'] == 'make')
df_plot_miss = df_plot_make.copy()
df_plot_make.loc[~bool_make, 'Streak Count'] = 0
df_plot_miss.loc[bool_make, 'Streak Count'] = 0

# Plot
sns.barplot(x=df_plot_make.index, y=df_plot_make['Streak Count'], color='#f85a40', label='Make') ;
sns.barplot(x=df_plot_miss.index, y=df_plot_miss['Streak Count'] * -1, color='#4298b5', label='Miss') ;

# Add cosmetics
plt.title('Kevin Durant 3p shoot throughout NBA Playoff 2018', fontname='Monospace', fontsize='20', fontweight='bold') ;
ax.set_xticklabels([])
plt.legend() ;


# In[ ]:


# Plotting shoot progress
rcParams['figure.figsize'] = [15,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
fig, ax = plt.subplots()

# Prep plot
bool_shot = (df_streaks['shot_type'] == '3PT')
bool_player = (df_streaks['player_name'] == 'Klay Thompson')
df_plot_make = df_streaks[(bool_shot & bool_player)].reset_index()
bool_make = (df_plot_make['result'] == 'make')
df_plot_miss = df_plot_make.copy()
df_plot_make.loc[~bool_make, 'Streak Count'] = 0
df_plot_miss.loc[bool_make, 'Streak Count'] = 0

# Plot
sns.barplot(x=df_plot_make.index, y=df_plot_make['Streak Count'], color='#f85a40', label='Make') ;
sns.barplot(x=df_plot_miss.index, y=df_plot_miss['Streak Count'] * -1, color='#4298b5', label='Miss') ;

# Add cosmetics
plt.title('Klay Thompson 3p shoot throughout NBA Playoff 2018', fontname='Monospace', fontsize='20', fontweight='bold') ;
ax.set_xticklabels([])
plt.legend() ;


# In[ ]:


# Plotting shoot progress
rcParams['figure.figsize'] = [15,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
fig, ax = plt.subplots()

# Prep plot
bool_shot = (df_streaks['shot_type'] == '3PT')
bool_player = (df_streaks['player_name'] == 'Stephen Curry')
df_plot_make = df_streaks[(bool_shot & bool_player)].reset_index()
bool_make = (df_plot_make['result'] == 'make')
df_plot_miss = df_plot_make.copy()
df_plot_make.loc[~bool_make, 'Streak Count'] = 0
df_plot_miss.loc[bool_make, 'Streak Count'] = 0

# Plot
sns.barplot(x=df_plot_make.index, y=df_plot_make['Streak Count'], color='#f85a40', label='Make') ;
sns.barplot(x=df_plot_miss.index, y=df_plot_miss['Streak Count'] * -1, color='#4298b5', label='Miss') ;

# Add cosmetics
plt.title('Stephen Curry 3p shoot throughout NBA Playoff 2018', fontname='Monospace', fontsize='20', fontweight='bold') ;
ax.set_xticklabels([])
plt.legend() ;


# Well turn out all three of them have multiple drought on 3p on a certain time, with a maximum of at least 8 3p missed for them. That is quite horrendous but hey they still win it all :D. Hope you guys enjoy this and please give me a feedback for my wrongdoing, Cheers !

# In[ ]:




